import numpy as np
import pandas as pd # Pandas : データベースの操作
import matplotlib.pyplot as plt # Matplotlib : グラフの描画


def predict(n):
    # CSVファイルの読み込み
    df = pd.read_csv("testdata.csv")
    # データの抽出
    x = df['x']
    y = df['y']

    # 横軸をx,縦軸をyの散布図(scatter)をプロット
    # plt.scatter(x, y)
    # plt.show()
    # データの概要を表示
    # df.describe()


    # センタリング
    df_c = df - df.mean()

    # データの概要を表示
    # df_c.describe()

    # データの抽出
    x = df_c['x']
    y = df_c['y']

    # xとyの散布図をプロット
    # plt.scatter(x, y)
    # plt.show()

    # パラメータaの計算
    xx = x * x
    xy = x * y
    a = xy.sum() / xx.sum()

    # グラフへ描画
    # plt.scatter(x, y, label = 'y') # 実測値
    # plt.plot(x, a*x, label = 'y_hat', color = 'red') # 予測値
    # plt.legend()
    # plt.show()

    mean = df.mean()

    xc = n - mean['x']
    # 単回帰分析による予測
    yc = a * xc
    # 元のスケールの予測値
    y_hat = a * xc + mean['y']

    # 出力
    return print(int(y_hat))
#####################################

predict(40)
