请基于输入CG图，生成 {shot_count} 张差分CG分镜脚本。
可选剧情描述：{story_desc}
要求：变化平滑、每镜头主体一致但动作/表情/构图有递进变化。
原图已有正向prompt（用于保留lora等加载）：{base_prompt}
原图已有负向prompt：{base_negative}
请在构图变化描述基础上输出每个镜头的增量 prompt（也可完整prompt），最终会和原图prompt合并。
