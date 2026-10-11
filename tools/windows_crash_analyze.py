"""Offline x64 Windows minidump inspection using the local DbgHelp library.

No symbol downloads, process attachment, or system configuration changes.
Usage: python tools/windows_crash_analyze.py dump.dmp --output report.json
"""
import argparse
import ctypes as C
import json
import os
import struct
from pathlib import Path


class Dump:
    def __init__(self, path):
        self.data = Path(path).read_bytes()
        signature, _, count, directory = self.unpack('<IIII', 0)
        if signature != 0x504D444D:
            raise ValueError('Not a Windows minidump')
        self.streams = dict((kind, (size, offset)) for kind, size, offset in
                            (self.unpack('<III', directory + i * 12) for i in range(count)) if kind)
        if self.unpack('<H', self.streams[7][1])[0] != 9:
            raise ValueError('Only AMD64 dumps are supported')
        self.modules = []
        offset = self.streams[4][1]
        for i in range(self.unpack('<I', offset)[0]):
            base, size, checksum, timestamp, name = self.unpack('<QIIII', offset + 4 + i * 108)
            length = self.unpack('<I', name)[0]
            path = self.data[name + 4:name + 4 + length].decode('utf-16-le')
            self.modules.append(dict(base=base, size=size, path=path, timestamp=timestamp))
        self.memory = []
        if 5 in self.streams:
            offset = self.streams[5][1]
            for i in range(self.unpack('<I', offset)[0]):
                base, size, rva = self.unpack('<QII', offset + 4 + i * 16)
                self.memory.append((base, size, rva))
        if 9 in self.streams:
            offset = self.streams[9][1]
            count, rva = self.unpack('<QQ', offset)
            for i in range(count):
                base, size = self.unpack('<QQ', offset + 16 + i * 16)
                self.memory.append((base, size, rva))
                rva += size
        offset = self.streams[6][1]
        self.crash_thread = self.unpack('<I', offset)[0]
        code, flags, record, address, count = self.unpack('<IIQQI', offset + 8)
        self.exception = dict(code=hex(code), address=hex(address),
                              parameters=list(self.unpack('<' + 'Q' * min(count, 15), offset + 40)))
        context_size, context_rva = self.unpack('<II', offset + 160)
        self.context = self.data[context_rva:context_rva + context_size]
        self.threads = []
        self.images = {}
        offset = self.streams[3][1]
        for i in range(self.unpack('<I', offset)[0]):
            pos = offset + 4 + i * 48
            tid = self.unpack('<I', pos)[0]
            context_size, context_rva = self.unpack('<II', pos + 40)
            self.threads.append((tid, self.data[context_rva:context_rva + context_size]))

    def unpack(self, fmt, offset):
        return struct.unpack_from(fmt, self.data, offset)

    def read(self, address, size):
        for base, length, offset in self.memory:
            if base <= address < base + length:
                available = min(size, base + length - address)
                return self.data[offset + address - base:offset + address - base + available]
        module = self.module(address)
        if module:
            path = module['path']
            if path not in self.images:
                try:
                    data = Path(path).read_bytes()
                    pe = struct.unpack_from('<I', data, 60)[0]
                    count, timestamp = struct.unpack_from('<HI', data, pe + 6)
                    optional_size = struct.unpack_from('<H', data, pe + 20)[0]
                    if timestamp != module['timestamp']:
                        raise ValueError('Local image timestamp differs from dump')
                    sections = []
                    for i in range(count):
                        pos = pe + 24 + optional_size + i * 40
                        virtual_size, virtual, raw_size, raw = struct.unpack_from('<IIII', data, pos + 8)
                        sections.append((virtual, raw_size, raw))
                    self.images[path] = (data, sections)
                except (OSError, ValueError, struct.error):
                    self.images[path] = None
            image = self.images[path]
            if image:
                data, sections = image
                rva = address - module['base']
                for virtual, length, raw in sections:
                    if virtual <= rva < virtual + length:
                        pos = raw + rva - virtual
                        return data[pos:pos + min(size, virtual + length - rva)]
        return b''

    def module(self, address):
        return next((m for m in self.modules if m['base'] <= address < m['base'] + m['size']), None)


class SymbolInfo(C.Structure):
    _fields_ = [('SizeOfStruct', C.c_uint32), ('TypeIndex', C.c_uint32), ('Reserved', C.c_uint64 * 2),
                ('Index', C.c_uint32), ('Size', C.c_uint32), ('ModBase', C.c_uint64),
                ('Flags', C.c_uint32), ('Value', C.c_uint64), ('Address', C.c_uint64),
                ('Register', C.c_uint32), ('Scope', C.c_uint32), ('Tag', C.c_uint32),
                ('NameLen', C.c_uint32), ('MaxNameLen', C.c_uint32), ('Name', C.c_char * 1)]


class Symbols:
    def __init__(self, dump):
        self.dump = dump
        self.lib = C.WinDLL('dbghelp', use_last_error=True)
        self.handle = C.c_void_p(0x12345)
        self.lib.SymInitializeW.argtypes = [C.c_void_p, C.c_wchar_p, C.c_int]
        self.lib.SymInitializeW.restype = C.c_int
        self.lib.SymSetOptions.argtypes = [C.c_uint32]
        # Undecorate, deferred loads, fail critical errors, and suppress prompts.
        self.lib.SymSetOptions(0x2 | 0x4 | 0x200 | 0x80000)
        if not self.lib.SymInitializeW(self.handle, '', False):
            raise C.WinError(C.get_last_error())
        self.lib.SymLoadModuleExW.argtypes = [C.c_void_p, C.c_void_p, C.c_wchar_p, C.c_wchar_p,
                                             C.c_uint64, C.c_uint32, C.c_void_p, C.c_uint32]
        self.lib.SymLoadModuleExW.restype = C.c_uint64
        for m in dump.modules:
            try:
                exists = Path(m['path']).is_file()
            except OSError:
                exists = False
            if exists:
                self.lib.SymLoadModuleExW(self.handle, None, m['path'], None, m['base'], m['size'], None, 0)
        self.lib.SymFromAddr.argtypes = [C.c_void_p, C.c_uint64, C.POINTER(C.c_uint64), C.c_void_p]
        self.lib.SymFromAddr.restype = C.c_int
        self.lib.StackWalk64.argtypes = [C.c_uint32, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
                                        C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
        self.lib.StackWalk64.restype = C.c_int
        self.lib.SymFunctionTableAccess64.argtypes = [C.c_void_p, C.c_uint64]
        self.lib.SymFunctionTableAccess64.restype = C.c_void_p
        self.lib.SymGetModuleBase64.argtypes = [C.c_void_p, C.c_uint64]
        self.lib.SymGetModuleBase64.restype = C.c_uint64
        reader = C.WINFUNCTYPE(C.c_int, C.c_void_p, C.c_uint64, C.c_void_p, C.c_uint32,
                               C.POINTER(C.c_uint32))

        def read_memory(handle, address, buffer, size, read_count):
            data = dump.read(address, size)
            read_count[0] = len(data)
            if data:
                C.memmove(buffer, data, len(data))
            return bool(data)
        self.reader = reader(read_memory)

    def describe(self, address):
        module = self.dump.module(address)
        result = dict(address=hex(address), module=Path(module['path']).name if module else None,
                      offset=hex(address - module['base']) if module else None)
        buffer = C.create_string_buffer(C.sizeof(SymbolInfo) + 2048)
        info = SymbolInfo.from_buffer(buffer)
        info.SizeOfStruct = C.sizeof(SymbolInfo)
        info.MaxNameLen = 2048
        displacement = C.c_uint64()
        if self.lib.SymFromAddr(self.handle, address, C.byref(displacement), buffer):
            result.update(symbol=C.string_at(C.addressof(buffer) + SymbolInfo.Name.offset, info.NameLen)
                          .decode('utf-8', errors='replace'), displacement=hex(displacement.value))
        return result

    def stack(self, context, thread):
        if len(context) < 256:
            return []
        rsp, rbp, rip = (struct.unpack_from('<Q', context, off)[0] for off in (152, 160, 248))
        frame = C.create_string_buffer(4096)
        for offset, value in ((0, rip), (32, rbp), (48, rsp)):
            struct.pack_into('<Q', frame, offset, value)
            struct.pack_into('<I', frame, offset + 12, 3)
        ctx = C.create_string_buffer(context)
        frames = [self.describe(rip)]
        seen = set()
        for _ in range(60):
            if not self.lib.StackWalk64(0x8664, self.handle, C.c_void_p(thread), frame, ctx,
                                       self.reader, self.lib.SymFunctionTableAccess64,
                                       self.lib.SymGetModuleBase64, None):
                break
            address = struct.unpack_from('<Q', frame, 0)[0]
            stack = struct.unpack_from('<Q', frame, 48)[0]
            if not address or not self.dump.module(address) or (address, stack) in seen:
                break
            seen.add((address, stack))
            if address != rip or len(frames) > 1:
                frames.append(self.describe(address))
        return frames

    def stack_candidates(self, context):
        """Pointers on the stack, NOT an unwound call trace."""
        rsp = struct.unpack_from('<Q', context, 152)[0]
        data = self.dump.read(rsp, 16384)
        found = []
        for offset in range(0, len(data) - 7, 8):
            address = struct.unpack_from('<Q', data, offset)[0]
            if self.dump.module(address):
                found.append(dict(stack_offset=hex(offset), **self.describe(address)))
        return found[:120]

    def close(self):
        self.lib.SymCleanup.argtypes = [C.c_void_p]
        self.lib.SymCleanup(self.handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dump')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if os.name != 'nt':
        parser.error('Windows is required for DbgHelp')
    dump = Dump(args.dump)
    symbols = Symbols(dump)
    try:
        report = dict(dump=str(Path(args.dump).absolute()), exception=dump.exception,
                      crash_thread=dump.crash_thread, symbols='local DLL exports/PDB only; no symbol downloads',
                      frames=symbols.stack(dump.context, dump.crash_thread),
                      stack_address_candidates=symbols.stack_candidates(dump.context),
                      candidate_warning='Raw stack pointers are not an unwound call trace; nearby export symbols may be imprecise.',
                      threads=[dict(thread=tid, frames=symbols.stack(ctx, tid)) for tid, ctx in dump.threads])
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in report.items()
                          if k not in ('threads', 'stack_address_candidates')}, ensure_ascii=True, indent=2))
    finally:
        symbols.close()


if __name__ == '__main__':
    main()
