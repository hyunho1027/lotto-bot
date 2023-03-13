import os, pathlib, requests

from bs4 import BeautifulSoup as bs
import pandas as pd

from util import get_config

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

def get_last_round() -> int:
    res = requests.get(URL)
    soup = bs(res.text, "html.parser")
    last_round = int(soup.find("div", "win_result").find("h4").find("strong").text.replace("회",""))
    return last_round

def collect():
    root_path = pathlib.Path(__file__).parent.resolve()
    cfg = get_config()
    output_path = os.path.join(root_path, cfg["data_path"], cfg["file_name"])
    try:
        df = pd.read_csv(output_path)
    except:
        df = pd.DataFrame(columns=["w1", "w2", "w3", "w4", "w5", "w6", "b"])

    last_round = get_last_round()
    for i in range(len(df), last_round):
        round = i+1
        nums = get_nums(round)
        df.loc[i] = nums
        with open(os.path.join(root_path, cfg["log_path"], "log.txt"), "a") as f:
            f.write(f"[REAL] {round}: {nums}\n")
    df.to_csv(output_path, index=False)

if __name__=="__main__":
    collect()