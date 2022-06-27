import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


def send_email(to, subject, content, attachment_filename_with_extension, pwd):
    """return True if sent, False if error"""
    email_sender = 'email'
    email_recipient = to
    msg = MIMEMultipart()
    msg['From'] = email_sender
    msg['To'] = email_recipient
    msg['Subject'] = subject

    email_message = content

    msg.attach(MIMEText(email_message, 'plain'))

    with open(attachment_filename_with_extension, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {attachment_filename_with_extension}",
    )

    # Add attachment to message
    msg.attach(part)

    try:
        server = smtplib.SMTP('smtp-mail.outlook.com',
                              port=587)
        server.ehlo()
        server.starttls()
        server.login(email_sender, pwd)
        text = msg.as_string()
        server.sendmail(email_sender, email_recipient, text)
        server.quit()
        return True
    except:
        print("SMPT server connection error")
        return False
