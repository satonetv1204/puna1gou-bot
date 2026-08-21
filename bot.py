import os
import json
import uuid
import asyncio
import discord
import pytesseract
import re
import cv2
import numpy as np

from PIL import Image
from io import BytesIO
from flask import Flask
from threading import Thread
from collections import Counter
from discord import app_commands
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# =====================
# Render用Webサーバー
# =====================

app = Flask(__name__)


@app.route("/")
def home():
    return "Bot is alive!"


def run_web():
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))


Thread(target=run_web, daemon=True).start()

# =====================
# TOKEN
# =====================

TOKEN = os.getenv("TOKEN")

# =====================
# WindowsのみTesseractパス設定
# =====================

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

# =====================
# Discord設定
# =====================

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# =====================
# JST
# =====================

JST = ZoneInfo("Asia/Tokyo")

# =====================
# 予約ファイル
# =====================

SCHEDULE_FILE = "schedules.json"

if not os.path.exists(SCHEDULE_FILE):
    with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# =====================
# OCR設定
# =====================

# 8桁対応：5〜9桁まで許可（上限 999,999,999）
MIN_DIGITS = 5
MAX_DIGITS = 9
MIN_VALUE = 10_000
MAX_VALUE = 999_999_999

# 単語単位の信頼度がこれ未満のものは捨てる（ゴミ検出対策）
MIN_CONF = 40

# 何行分の数値を期待するか
EXPECTED_ROWS = 5

OCR_CONFIG = "--psm 6 -c tessedit_char_whitelist=0123456789"


def preprocess(img: Image.Image, variant: int) -> Image.Image:
    """左列（与えたダメージ）の数値領域を切り出して2値化する。
    variantごとに閾値処理を変え、多数決で誤読を潰す。"""
    w, h = img.size

    # 左列の数値部分だけを切り出す
    img = img.crop((int(w * 0.25), 0, int(w * 0.52), h))

    # 高品質リサンプリングで4倍に拡大（5/9の潰れ対策）
    img = img.resize((img.width * 4, img.height * 4), Image.LANCZOS)

    g = np.array(img.convert("L"))

    if variant == 0:
        # Otsuの自動閾値
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif variant == 1:
        # やや高めの固定閾値
        _, b = cv2.threshold(g, 150, 255, cv2.THRESH_BINARY)
    else:
        # ノイズ除去 + 低めの固定閾値
        g = cv2.medianBlur(g, 3)
        _, b = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY)

    return Image.fromarray(b)


def ocr_rows(img: Image.Image):
    """image_to_dataで数値を行位置・信頼度つきで抽出する。"""
    data = pytesseract.image_to_data(
        img,
        config=OCR_CONFIG,
        output_type=pytesseract.Output.DICT,
    )

    rows = []

    for i, txt in enumerate(data["text"]):
        t = txt.strip()
        try:
            conf = int(float(data["conf"][i]))
        except (ValueError, TypeError):
            continue

        if not re.fullmatch(rf"\d{{{MIN_DIGITS},{MAX_DIGITS}}}", t):
            continue

        val = int(t)

        if not (MIN_VALUE <= val <= MAX_VALUE):
            continue

        if conf < MIN_CONF:
            continue

        rows.append((data["top"][i], val, conf))

    return rows


def merge_rows(all_rows, row_tol=100):
    """複数パスの検出結果をy座標で行グルーピングし、
    行ごとに多数決（同数なら信頼度最大）で値を決める。"""
    groups = []  # [[y, [(val, conf), ...]], ...]

    for y, val, conf in all_rows:
        for g in groups:
            if abs(g[0] - y) <= row_tol:
                g[1].append((val, conf))
                break
        else:
            groups.append([y, [(val, conf)]])

    result = []

    for y, cands in sorted(groups):
        counts = Counter(v for v, c in cands)
        best = max(
            counts,
            key=lambda v: (
                counts[v],
                max(c for vv, c in cands if vv == v),
            ),
        )
        agree = counts[best] / len(cands)
        result.append({"y": y, "value": best, "agree": agree, "cands": cands})

    return result


def analyze_image(img: Image.Image):
    """3パスOCR + 多数決。同期関数（スレッドで実行する）。"""
    all_rows = []

    for v in range(3):
        p = preprocess(img.copy(), v)
        rows = ocr_rows(p)
        print(f"==== OCR variant {v} ====")
        print(rows)
        all_rows += rows

    merged = merge_rows(all_rows)

    print("最終:", [(m["value"], round(m["agree"], 2)) for m in merged])

    return merged

# =====================
# 予約監視ループ
# =====================

# プロセス内での二重送信ガード
sent_ids = set()


async def schedule_loop():
    await client.wait_until_ready()

    while not client.is_closed():

        try:

            with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
                schedules = json.load(f)

            now = datetime.now(JST)

            remaining = []

            for s in schedules:

                send_time = datetime.strptime(
                    s["time"],
                    "%Y-%m-%d %H:%M"
                ).replace(tzinfo=JST)

                if now >= send_time:

                    # 同じ予約を二度送らない
                    if s["id"] in sent_ids:
                        continue

                    sent_ids.add(s["id"])

                    channel = client.get_channel(s["channel_id"])

                    if channel:
                        await channel.send(s["message"])

                else:
                    remaining.append(s)

            with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    remaining,
                    f,
                    ensure_ascii=False,
                    indent=4
                )

        except Exception as e:
            print("予約エラー:", e)

        await asyncio.sleep(30)

# =====================
# 起動時
# =====================

# on_readyは再接続のたびに呼ばれるので、
# タスク生成とコマンド同期は初回だけにする
_started = False


@client.event
async def on_ready():

    global _started

    if not _started:
        _started = True

        await tree.sync()

        client.loop.create_task(schedule_loop())

    print(f"ログインした: {client.user}")

# =====================
# 予約コマンド
# =====================


@tree.command(
    name="予約",
    description="メッセージを予約送信する"
)
@app_commands.describe(
    日後="何日後か",
    時間="送信時間 (例: 22:00)",
    メッセージ="送信するメッセージ"
)
async def reserve(
    interaction: discord.Interaction,
    日後: int,
    時間: str,
    メッセージ: str
):

    try:

        hour, minute = map(int, 時間.split(":"))

        send_time = datetime.now(JST) + timedelta(days=日後)

        send_time = send_time.replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0
        )

        schedule_data = {
            "id": str(uuid.uuid4())[:8],
            "channel_id": interaction.channel.id,
            "message": メッセージ,
            "time": send_time.strftime("%Y-%m-%d %H:%M")
        }

        with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
            schedules = json.load(f)

        schedules.append(schedule_data)

        with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                schedules,
                f,
                ensure_ascii=False,
                indent=4
            )

        await interaction.response.send_message(
            f"予約したぷな〜！\n"
            f"ID: {schedule_data['id']}\n"
            f"日時: {send_time.strftime('%Y/%m/%d %H:%M')}\n"
            f"メッセージ: {メッセージ}",
            ephemeral=True
        )

    except Exception as e:

        await interaction.response.send_message(
            f"エラーぷな〜！\n{e}",
            ephemeral=True
        )

# =====================
# 予約一覧
# =====================


@tree.command(
    name="予約一覧",
    description="現在の予約一覧を見る"
)
async def schedule_list(interaction: discord.Interaction):

    with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
        schedules = json.load(f)

    if not schedules:

        await interaction.response.send_message(
            "予約はありません…ぷな…",
            ephemeral=True
        )

        return

    text = ""

    for s in schedules:

        text += (
            f"ID: {s['id']}\n"
            f"日時: {s['time']}\n"
            f"内容: {s['message']}\n\n"
        )

    await interaction.response.send_message(
        text,
        ephemeral=True
    )

# =====================
# 予約削除
# =====================


@tree.command(
    name="予約削除",
    description="予約を削除する"
)
@app_commands.describe(
    id="予約ID"
)
async def schedule_delete(
    interaction: discord.Interaction,
    id: str
):

    with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
        schedules = json.load(f)

    new_schedules = []

    deleted = False

    for s in schedules:

        if s["id"] == id:
            deleted = True
        else:
            new_schedules.append(s)

    with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            new_schedules,
            f,
            ensure_ascii=False,
            indent=4
        )

    if deleted:

        await interaction.response.send_message(
            f"予約 {id} を削除しました…ぷな！",
            ephemeral=True
        )

    else:

        await interaction.response.send_message(
            "そのIDは見つかりませんでした…ぷな…",
            ephemeral=True
        )

# =====================
# メイン処理
# =====================


@client.event
async def on_message(message):

    if message.author.bot:
        return

    if message.content in ["!使い方", "!help"]:

        await message.channel.send(
            "は...はじめましてぇ…っ\n"
            "社畜系看護師のぷないぷない、と申します…！\n"
            "ダメージレポートの画像を貼ってもらえれば…\n"
            "合計ダメージ、計算しますっ…ぷな！"
        )

        return

    if message.attachments:

        for attachment in message.attachments:

            if attachment.filename.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):

                # discord.py組み込みのreadを使う（requests不要）
                data = await attachment.read()

                img = Image.open(BytesIO(data))

                # OCRは重いのでスレッドに逃がす
                # （イベントループを止めない＝Botの反応が遅くならない）
                merged = await asyncio.to_thread(analyze_image, img)

                # 上位EXPECTED_ROWS行だけ使う（画面上の並び順のまま）
                merged = merged[:EXPECTED_ROWS]

                values = [m["value"] for m in merged]

                if len(values) == EXPECTED_ROWS:

                    total = sum(values)

                    formula = " + ".join(f"{n:,}" for n in values)

                    # OCRパス間で結果が割れた行があれば注意書き
                    shaky = [
                        f"{m['value']:,}"
                        for m in merged
                        if m["agree"] < 1.0
                    ]

                    warn = ""
                    if shaky:
                        warn = (
                            "\n⚠ "
                            + "、".join(shaky)
                            + " は読み取りが揺れたので確認してほしいぷな…"
                        )

                    await message.channel.send(
                        f"{formula}\n"
                        f"= {total:,}ぷな～{warn}"
                    )

                else:
                    # 5行そろわない＝ダメージレポートではない可能性が
                    # 高いので、何も送らずログだけ残す
                    print(f"読み取り{len(values)}行のためスキップ")

# =====================
# 起動
# =====================

client.run(TOKEN)
