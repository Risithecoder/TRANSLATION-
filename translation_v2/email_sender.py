"""
email_sender.py — Send translated DOCX via Gmail SMTP.

Uses Python's built-in smtplib + email.mime modules (no pip install needed).
Requires two environment variables:
    SMTP_EMAIL    — the Gmail address to send from
    SMTP_PASSWORD — a Gmail App Password (NOT the regular password)
"""

import os
import socket
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


# ── Constants ─────────────────────────────────────────────────────────────────
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
FIXED_CC = "vinay.ramakrishna@oliveboard.in"
DOMAIN = "@oliveboard.in"


def send_translation_email(
    username: str,
    docx_path: str,
    language: str,
    question_count: int,
    original_filename: str,
) -> dict:
    """
    Send the translated DOCX to {username}@oliveboard.in with CC to Vinay.

    Args:
        username:          Oliveboard username (without @oliveboard.in)
        docx_path:         Absolute path to the translated .docx file
        language:          Target language (e.g. "Hindi")
        question_count:    Number of questions translated
        original_filename: Original uploaded filename (e.g. "SSC_Paper.docx")

    Returns:
        dict with "ok": True on success, or raises RuntimeError on failure.
    """
    sender_email = os.environ.get("SMTP_EMAIL", "")
    sender_password = os.environ.get("SMTP_PASSWORD", "")

    if not sender_email or not sender_password:
        raise RuntimeError(
            "Email credentials not configured. "
            "Set SMTP_EMAIL and SMTP_PASSWORD in your .env file."
        )

    to_email = f"{username}{DOMAIN}"
    docx_filename = os.path.basename(docx_path)

    # ── Build the email ───────────────────────────────────────────────────
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Cc"] = FIXED_CC
    msg["Subject"] = f"Translation Complete — {language} — {original_filename}"

    body = (
        f"Hi,\n\n"
        f"Your translation is ready!\n\n"
        f"  • Language: {language}\n"
        f"  • Questions translated: {question_count}\n"
        f"  • Original file: {original_filename}\n\n"
        f"The translated document is attached.\n\n"
        f"— Oliveboard TranslateLab"
    )
    msg.attach(MIMEText(body, "plain"))

    # ── Attach the DOCX ───────────────────────────────────────────────────
    if not os.path.exists(docx_path):
        raise RuntimeError(f"DOCX file not found: {docx_path}")

    with open(docx_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f"attachment; filename={docx_filename}",
    )
    msg.attach(part)

    # ── Send via Gmail SMTP ───────────────────────────────────────────────
    recipients = [to_email, FIXED_CC]

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipients, msg.as_string())
        print(f"[email_sender] ✅ Email sent to {to_email} (CC: {FIXED_CC})")
        return {"ok": True, "to": to_email, "cc": FIXED_CC}
    except smtplib.SMTPAuthenticationError:
        raise RuntimeError(
            "Gmail authentication failed. Check your SMTP_EMAIL and "
            "SMTP_PASSWORD (must be a Gmail App Password)."
        )
    except (socket.timeout, smtplib.SMTPConnectError, ConnectionRefusedError, OSError) as e:
        print(f"[email_sender] ❌ Connection error: {e}")
        raise RuntimeError(
            "Cannot connect to the mail server. Check your internet "
            "connection and ensure port 587 is not blocked."
        )
    except Exception as e:
        print(f"[email_sender] ❌ Unexpected error: {e}")
        raise RuntimeError("Failed to send email due to an internal server error.")
