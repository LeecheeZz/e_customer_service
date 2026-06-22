import os
import tempfile

def replace_text_in_jsonl_inplace(
    file_path: str,
    old: str = " 具体以订单页面和平台规则为准。",
    new: str = "具体以订单页面和平台规则为准。"
) -> None:
    dir_name = os.path.dirname(file_path) or "."
    
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=dir_name
    ) as tmp:
        tmp_path = tmp.name

        with open(file_path, "r", encoding="utf-8") as fin:
            for line in fin:
                tmp.write(line.replace(old, new))

    os.replace(tmp_path, file_path)

if __name__ == '__main__':
    replace_text_in_jsonl_inplace("train_sft.jsonl")