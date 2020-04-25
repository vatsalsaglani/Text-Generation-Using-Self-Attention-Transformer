from slack_webhook import Slack

from slack_config import slack_urls


def send_slack(url, text):
    slack = Slack(url=url)
    slack.post(text=text)
