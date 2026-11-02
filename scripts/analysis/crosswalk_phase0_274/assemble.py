import io
parts = [open("post_head.md", encoding="utf-8").read(), open("table.md", encoding="utf-8").read(),
         open("post_mid.md", encoding="utf-8").read(), open("footprint.md", encoding="utf-8").read(),
         open("post_tail.md", encoding="utf-8").read()]
open("post.md", "w", encoding="utf-8").write("\n".join(parts))
print("post.md", sum(len(p) for p in parts), "chars")
