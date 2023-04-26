from config import *


def setup_email_pass():
    try:
        conn = smtplib.SMTP_SSL('smtp.mail.yahoo.com', 465)
        conn.ehlo()
        conn.login(EMAIL_ADDRESS, YAHOO_PASSCODE)
        conn.quit()

    except socket.error as e:
        logging.info("Could not connect to mail")


def setup_text_pass():
    try:
        client = Client(TWILIO_ACC, TWILIO_AUTH_KEY)
    except:
        logging.info("Could not connect to text")

    return client


def send_email(subject='Important', content='testing', footer='~George'):
    conn = smtplib.SMTP_SSL('smtp.mail.yahoo.com', 465)
    conn.ehlo()
    conn.login(EMAIL_ADDRESS, YAHOO_PASSCODE)
    conn.sendmail(EMAIL_ADDRESS, EMAIL_ADDRESS, subject + content + footer)
    conn.quit()


def send_message(host, message="Hello World"):
    host.messages.create(to=MY_NUMBER, from_=TWILIO_PHONE_NUMBER, body=message)


