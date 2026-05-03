from huggingface_hub import snapshot_download

print("FFHQ Part1~Part4 다운로드 시작...")

snapshot_download(
    repo_id="marcosv/ffhq-dataset",
    repo_type="dataset",
    allow_patterns=["Part1/*", "Part2/*", "Part3/*", "Part4/*"],
    local_dir="C:/dataset/ffhq"
)

print("다운로드 완료!")
print("저장 위치: C:/dataset/ffhq")