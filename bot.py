```python
import os
import json
import uuid
import asyncio
import discord
import pytesseract
import requests
import re
import cv2
import numpy as np

from PIL import Image
from io import BytesIO
from flask import Flask
from threading import Thread
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
    app.run(host="0.0.0.0", port=10000)


Thread(target=run_web).start()

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

# ファイルが無ければ作成
if not os.path.exists(SCHEDULE_FILE):
    with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# =====================
# 数字抽出
# =====================


def parse_numbers(text):
    raw = re.findall(r"\d+", text)

    numbers = []

    for n in raw:
        if 5 <= len(n) <= 8:
            val = int(n)

            if 10000 <= val <= 10000000:
                numbers.append(val)

    numbers = list(set(numbers))
    numbers = sorted(numbers, reverse=True)

    return numbers

# =====================
# 通常OCR
# =====================


def extract_main(img):
    w, h = img.size

    img = img.crop((0, 0, w // 2, h))

    img = img.crop((
        int(w * 0.35),
        0,
        int(w * 0.95),
        h
    ))

    img = img.resize((img.width * 3, img.height * 3))
    img = img.convert("L")
    img = img.point(lambda x: 0 if x < 130 else 255)

    text = pytesseract.image_to_string(
        img,
        config="--psm 6 -c tessedit_char_whitelist=0123456789"
    )

    print("==== 通常OCR ====")
    print(text)

    return parse_numbers(text)

# =====================
# 再抽出OCR
# =====================


def extract_retry(img):
    w, h = img.size

    img = img.crop((0, 0, w // 2, h))

    img = img.crop((
        int(w * 0.2),
        0,
        int(w * 0.95),
        h
    ))

    img = img.resize((img.width * 2, img.height * 2))
    img = img.convert("L")

    img_np = np.array(img)

    img_np = cv2.medianBlur(img_np, 3)

    _, img_np = cv2.threshold(
        img_np,
        180,
        255,
        cv2.THRESH_BINARY
    )

    img = Image.fromarray(img_np)

    text = pytesseract.image_to_string(
        img,
        config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"
    )

    print("==== 再抽出OCR ====")
    print(text)

    return parse_numbers(text)

# =====================
# 予約監視ループ
# =====================


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


@client.event
async def on_ready():

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

                response = requests.get(attachment.url)

                img = Image.open(BytesIO(response.content))

                # 通常
                numbers_main = extract_main(img)

                print("通常:", numbers_main)

                # 再抽出
                numbers_retry = []

                if len(numbers_main) < 5:

                    print("⚠️ 再抽出発動")

                    numbers_retry = extract_retry(img)

                    print("再抽出:", numbers_retry)

                # 統合
                numbers = list(
                    set(numbers_main + numbers_retry)
                )

                numbers = sorted(
                    numbers,
                    reverse=True
                )[:5]

                print("最終:", numbers)

                if len(numbers) == 5:

                    total = sum(numbers)

                    await message.channel.send(
                        f"合計ダメージ: {total:,}ぷな～"
                    )

# =====================
# 起動
# =====================

client.run(TOKEN)