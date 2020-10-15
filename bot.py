from telegram.ext import Updater

from settings import BOT_CHAT, BOT_TOKEN


class BotNotifier:
    def __init__(self):
        self.updater = Updater(token=BOT_TOKEN)

    def send(self, msg):
        self.updater.bot.send_message(BOT_CHAT, msg)
