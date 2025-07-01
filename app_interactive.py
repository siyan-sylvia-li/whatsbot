import app_official

app_official.twilio_client = None


if __name__ == "__main__":
    BODY_TEMPLATE = {
        "From": "007",
        "MessageType": "text",
        "Body": "The name is Bond. James Bond."
    }

    while True:
        user_input = input("USER >> ")
        body_t = BODY_TEMPLATE.copy()
        if user_input == "PING":
            body_t["MessageType"] = "button"
        else:
            body_t["Body"] = user_input
        twilio_client = None
        app_official.handle_whatsapp_message(body_t)