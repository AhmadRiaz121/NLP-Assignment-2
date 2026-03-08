# Conversational Flow Design: Hotel Front-Desk Virtual Assistant

---

## 1. High-Level System Flow

```
User Input
    │
    ▼
┌─────────────────────┐
│   Intent Detection   │  ◄── NLU / Classification Engine
└─────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   Intent Router                     │
│                                                     │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │Reservation │  │  Check-In /  │  │Room Service │  │
│  │            │  │  Check-Out   │  │             │  │
│  └────────────┘  └──────────────┘  └─────────────┘  │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │   FAQs     │  │  Complaint   │  │ Escalation  │  │
│  │            │  │  Handling    │  │(Human Agent)│  │
│  └────────────┘  └──────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
    │
    ▼
Response Generation (LLM Engine)
    │
    ▼
User Output (Streamed via WebSocket)
```

---

## 2. Flow 1: Room Reservation

```
[START]
    │
    ▼
Greet User & Ask for Check-in Date
    │
    ▼
Collect Check-out Date
    │
    ▼
Ask Room Type Preference
    │
    ▼
Check Availability in PMS
    │
    ├── [Available] ──────────► Show Room Options & Pricing
    │                                       │
    │                                       ▼
    │                               Guest Confirms Choice
    │                                       │
    │                                       ▼
    │                               Collect Guest Details
    │                               (Name, Email, Phone)
    │                                       │
    │                                       ▼
    │                               Process Payment Info
    │                                       │
    │                                       ▼
    │                               Confirm Booking
    │                               Send Confirmation Email
    │                                       │
    │                                       ▼
    │                                     [END]
    │
    └── [Not Available] ────────► Suggest Alternate Dates / Room Types
                                              │
                                              ├── Guest Accepts ──► Back to "Show Room Options"
                                              └── Guest Declines ──► Offer Human Assistance ──► [END]
```

---

## 3. Flow 2: Check-In

```
[START]
    │
    ▼
Ask for Reservation ID or Guest Name
    │
    ▼
Lookup Reservation in PMS
    │
    ├── [Found] ─────────────► Display Booking Summary
    │                                   │
    │                                   ▼
    │                           Confirm Guest Identity
    │                           (DOB / ID Number)
    │                                   │
    │                                   ▼
    │                           Assign Room & Issue Digital Key
    │                                   │
    │                                   ▼
    │                           Provide Hotel Info
    │                           (Breakfast, Amenities, Services)
    │                                   │
    │                                   ▼
    │                           Ask if Further Assistance Needed
    │                                   │
    │                                   ▼
    │                                 [END]
    │
    └── [Not Found] ──────────► Ask Guest to Re-enter Details
                                        │
                                        ├── [Found on Retry] ──► Continue Flow Above
                                        └── [Still Not Found] ──► Escalate to Human ──► [END]
```

---

## 4. Flow 3: Check-Out

```
[START]
    │
    ▼
Ask for Room Number / Reservation ID
    │
    ▼
Retrieve Itemized Bill from PMS
    │
    ▼
Display Bill to Guest
    │
    ▼
Guest Reviews Bill
    │
    ├── [Accepts Bill] ───────► Process Payment
    │                                   │
    │                                   ▼
    │                           Send Receipt via Email
    │                                   │
    │                                   ▼
    │                           Request Feedback / Review
    │                                   │
    │                                   ▼
    │                           Farewell Message ──► [END]
    │
    └── [Disputes Charge] ────► Log Dispute
                                        │
                                        ▼
                                Offer Resolution Options
                                        │
                                        ├── Process Adjusted Bill ──► Payment ──► [END]
                                        └── Escalate to Billing Manager ──► [END]
```

---

## 5. Flow 4: Room Service

```
[START]
    │
    ▼
Identify In-House Guest (Room Number)
    │
    ▼
Ask for Service Type
    │
    ├── [Food Order] ──────────► Show Menu
    │                                 │
    │                                 ▼
    │                           Guest Selects Items
    │                                 │
    │                                 ▼
    │                           Confirm Order & Total
    │                                 │
    │                                 ▼
    │                           Charge to Room Bill?
    │                                 │
    │                                 ▼
    │                           Place Order → Provide ETA ──► [END]
    │
    ├── [Housekeeping] ───────► Ask Preferred Time
    │                                 │
    │                                 ▼
    │                           Schedule & Confirm ──► [END]
    │
    ├── [Amenities] ──────────► Ask What's Needed
    │                                 │
    │                                 ▼
    │                           Dispatch → Confirm ETA ──► [END]
    │
    └── [Maintenance] ────────► Log Issue
                                      │
                                      ▼
                                Dispatch Maintenance Team
                                      │
                                      ▼
                                Offer Interim Solutions ──► [END]
```

---

## 6. Flow 5: FAQ Handling

```
[START]
    │
    ▼
Detect FAQ Intent from User Query
    │
    ▼
Match Query Against Knowledge Base
    │
    ├── [High Confidence Match] ──► Provide Direct Answer
    │                                       │
    │                                       ▼
    │                               Ask if More Help Needed ──► [END]
    │
    ├── [Low Confidence Match] ───► Provide Best Match + Clarifying Options ──► [END]
    │
    └── [No Match] ───────────────► Apologize
                                          │
                                          ▼
                                    Suggest Calling Front Desk
                                    OR Escalate to Human ──► [END]
```

---

## 7. Flow 6: Complaint Handling

```
[START]
    │
    ▼
Acknowledge Complaint with Empathy
    │
    ▼
Collect Room Number & Complaint Details
    │
    ▼
Log Complaint in System
    │
    ▼
Assess Severity Level
    │
    ├── [Low Severity] ────────► Provide Immediate Resolution
    │                                   │
    │                                   ▼
    │                           Confirm Resolution with Guest ──► [END]
    │
    ├── [Medium Severity] ────► Dispatch Relevant Department
    │                                   │
    │                                   ▼
    │                           Provide ETA + Offer Interim Solution ──► [END]
    │
    └── [High Severity /
         Guest Requests Human] ──► Immediate Escalation
                                          │
                                          ▼
                                    Transfer to Human Agent
                                    with Full Conversation Context ──► [END]
```

---

## 8. General Escalation Flow

```
[Escalation Triggered]
    │
    ▼
Inform Guest: "Connecting you with a team member..."
    │
    ▼
Summarize Conversation Context for Human Agent
    │
    ▼
Transfer to Available Human Agent
    │
    ├── [Agent Available] ──────► Seamless Handoff ──► [END]
    │
    └── [No Agent Available] ───► Offer Callback / Email Follow-up
                                          │
                                          ▼
                                    Log Request ──► [END]
```

---

## 9. Conversation Design Principles

| Principle            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| **Clarity**          | Always confirm what the user said before taking action                      |
| **Empathy**          | Use warm, polite language especially during complaints                      |
| **Brevity**          | Keep responses concise; avoid information overload                          |
| **Fallback**         | Always provide a human escalation option                                    |
| **Context Retention**| Remember previous turns within the same session                             |
| **Confirmation**     | Confirm critical actions (bookings, payments) before executing              |
| **Error Recovery**   | Offer clear paths when input is unclear or invalid                          |

---

## 10. Slot-Filling Table (Key Intents)

| Intent            | Required Slots                                         | Optional Slots        |
|-------------------|--------------------------------------------------------|-----------------------|
| Room Reservation  | check_in_date, check_out_date, room_type, guest_name, email | special_requests |
| Check-In          | reservation_id OR guest_name                           | room_preferences      |
| Check-Out         | room_number OR reservation_id                          | feedback              |
| Room Service      | room_number, service_type                              | delivery_time         |
| Complaint         | room_number, complaint_description                     | preferred_resolution  |
| FAQ               | query_topic                                            | —                     |
