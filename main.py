from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
import datetime

app = FastAPI()

SESSIONS = {}

def estimate_damage_from_image(media_url: str):
    severity = "Moderate"
    estimated_cost_range = "$450–$1,200"
    return severity, estimated_cost_range

def get_appointment_slots(n: int = 3):
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    hours = [10, 13, 16]
    slots = []
    for h in hours:
        slot = tomorrow.replace(hour=h, minute=0, second=0, microsecond=0)
        if slot > now:
            slots.append(slot)
    return slots[:n]

@app.get("/")
def root():
    return {"status": "Backend is running!"}

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    from_number = form.get("From")
    body = (form.get("Body") or "").strip()
    media_url = form.get("MediaUrl0")

    reply = MessagingResponse()
    session = SESSIONS.get(from_number)

    if session and session.get("awaiting_time") and body in {"1", "2", "3"}:
        choice = int(body) - 1
        slots = session["slots"]
        if 0 <= choice < len(slots):
            chosen_slot = slots[choice]
            reply.message(
                f"Perfect! Your appointment is booked for "
                f"{chosen_slot.strftime('%a %b %d at %I:%M %p')}."
            )
            session["awaiting_time"] = False
            SESSIONS[from_number] = session
            return Response(content=str(reply), media_type="application/xml")

    if media_url:
        severity, estimate_range = estimate_damage_from_image(media_url)
        slots = get_appointment_slots()
        SESSIONS[from_number] = {"awaiting_time": True, "slots": slots}

        msg = (
            "🛠 Quick AI Estimate
"
            f"Damage Level: {severity}
"
            f"Estimated Cost: {estimate_range}

"
            "Reply with a number to book an appointment:
"
        )
        for i, s in enumerate(slots, 1):
            msg += f"{i}) {s.strftime('%a %b %d at %I:%M %p')}
"
        reply.message(msg)
        return Response(content=str(reply), media_type="application/xml")

    reply.message(
        "Thanks for contacting [Shop Name]! 👋

"
        "Please send a clear photo of the damage for an AI estimate."
    )
    return Response(content=str(reply), media_type="application/xml")
