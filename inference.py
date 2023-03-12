import time, pathlib, os, argparse

from playwright.sync_api import Playwright, sync_playwright
import pandas as pd
import torch
import mlflow

from common import get_config, Model

def buy(playwright: Playwright, user_id, user_pw, nums_list: list) -> None:
    # chrome 브라우저를 실행
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()

    # Open new page
    page = context.new_page()
    
    # 로그인
    page.goto("https://dhlottery.co.kr/user.do?method=login")
    page.click("[placeholder=\"아이디\"]")
    page.fill("[placeholder=\"아이디\"]", user_id)
    page.press("[placeholder=\"아이디\"]", "Tab")
    page.fill("[placeholder=\"비밀번호\"]", user_pw)
    page.press("[placeholder=\"비밀번호\"]", "Tab")
    with page.expect_navigation():
        page.press("form[name=\"jform\"] >> text=로그인", "Enter")
    
    # 페이지이동
    page.goto(url="https://ol.dhlottery.co.kr/olotto/game/game645.do")

    # 번호입력
    for nums in nums_list:
        for num in nums:
            page.click(f"label:has-text(\"{num}\")")
        page.click("text=확인")
    time.sleep(10)

    # 자동선택
    # page.click("text=자동선택")
    # page.select_option("select", str(COUNT))

    page.click("input:has-text(\"구매하기\")")
    page.click("text=확인 취소 >> input[type=\"button\"]")
    page.click("input[name=\"closeLayer\"]")
    # ---------------------

    context.close()
    browser.close()

def inference(user_id, user_pw):
    root_path = pathlib.Path(__file__).parent.resolve()
    cfg = get_config()

    z = pd.read_csv(os.path.join(root_path, cfg["dataset_path"], "z.csv")).values.squeeze()
    z = torch.FloatTensor(z)

    # TODO: get best model not latest model
    model = mlflow.search_registered_models(filter_string="name='lotto'")[0]
    model_version = model.latest_versions[0]
    model_uri = f"runs:/{model_version.run_id}/{cfg['model_path']}"
    net = mlflow.pytorch.load_model(model_uri)
    
    run = mlflow.get_run(model_version.run_id)
    window_size = int(run.data.params['window_size'])
    m = Model(cfg["input_dim"] * window_size, cfg["hidden_dim"], cfg["output_dim"])
    m.net = net

    # TODO: add transform module
    z = z[-cfg["input_dim"]*window_size:]

    # TODO: query to serving model    
    preds = m.predict(z.repeat(cfg["count"], 1))
    
    n = len(pd.read_csv(os.path.join(root_path, cfg["data_path"], cfg["file_name"])))
    with open(os.path.join(root_path, cfg["log_path"], "log.txt"), "a") as f:
        f.write(f"[PRED] {n+1}: {preds}\n")
    
    with sync_playwright() as playwright:
        buy(playwright, user_id, user_pw, preds)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id")
    parser.add_argument("--pw")
    args = parser.parse_args()

    inference(args.id, args.pw)