import smtplib
from email.mime.text import MIMEText
from .settings import settings


def verification_email_html(verify_link: str) -> str:
    return f"""
    <div style="font-family:Arial,Helvetica,sans-serif;line-height:1.6">
      <h2>Confirm your email</h2>
      <p>Click the button below to verify your email address:</p>
      <p><a href="{verify_link}" style="background:#086dd6;color:#fff;padding:10px 16px;border-radius:6px;text-decoration:none">Verify email</a></p>
      <p>If the button doesn't work, copy and paste this link into your browser:</p>
      <p><code>{verify_link}</code></p>
    </div>
    """


def reset_password_email_html(reset_link: str) -> str:
    """Return simple HTML for the reset password email."""
    return f"""
    <html>
      <body>
        <p>You requested a password reset.</p>
        <p>Click the link below to set a new password:</p>
        <p><a href="{reset_link}">Reset your password</a></p>
        <p>If you did not request this, you can ignore this email.</p>
      </body>
    </html>
    """



def send_email(to: str, subject: str, body: str):
    msg = MIMEText(body, "html")
    msg["Subject"] = subject
    msg["From"] = settings.mail_from
    msg["To"] = to

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as s:
        if settings.smtp_tls:
            s.starttls()
        s.login(settings.smtp_user, settings.smtp_password)
        s.send_message(msg)