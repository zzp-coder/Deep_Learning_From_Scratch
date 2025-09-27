from moviepy.editor import VideoFileClip, concatenate_videoclips

# 读入两个 mov 视频
clip1 = VideoFileClip("video1.mov")
clip2 = VideoFileClip("video2.mov")

# 把 clip2 调整成和 clip1 一致的分辨率 & 帧率
clip2 = clip2.resize(clip1.size).set_fps(clip1.fps)

# 顺序拼接（clip1 播完接着 clip2）
final_clip = concatenate_videoclips([clip1, clip2], method="compose")

# 输出结果
final_clip.write_videofile("output.mov", codec="libx264", audio_codec="aac")