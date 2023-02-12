import os, pathlib, requests

from bs4 import BeautifulSoup as bs
import pandas as pd

from common import get_config

URL = 'https://dhlottery.co.kr/gameResult.do?method=byWin'

def get_nums(drwNo: int):
    data = {"drwNo": drwNo}
    res = requests.post(URL, data=data)
    if res.status_code != 200:
        raise ConnectionError("requests failed")
    soup = bs(res.text, "html.parser")

    num_win = soup.find("div", "num win").find("p").find_all("span")
    num_bonus = soup.find("div", "num bonus").find("p").find_all("span")

    get_num = lambda x: int(x.text)
    try:
        nums = [get_num(x) for x in num_win + num_bonus]
    except:
        raise RuntimeError("drwNo is invalid")
    
    print(drwNo, ":", nums)
    return nums

def collect():
    root_path = pathlib.Path(__file__).parent.resolve()
    cfg = get_config()
    output_path = os.path.join(root_path, cfg["data_path"], cfg["file_name"])
    try:
        df = pd.read_csv(output_path)
    except:
        df = pd.DataFrame(columns=["w1", "w2", "w3", "w4", "w5", "w6", "b"])
    history_len = len(df)
    try:
        while True:
            nums = get_nums(history_len + 1)
            df.loc[history_len] = nums
            history_len += 1
            with open(os.path.join(root_path, cfg["log_path"], "log.txt"), "a") as f:
                f.write(f"[REAL] {history_len}: {nums}\n")
    except:
        df.to_csv(output_path, index=False)

if __name__=="__main__":
    collect()