import difflib

file1 = "google-out.md"
file2 = "llama-out.md"

with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

diff = difflib.unified_diff(
    lines1, lines2,
    fromfile=file1, tofile=file2,
    lineterm=""
)

for line in diff:
    print(line)