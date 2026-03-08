# Use-Case Description: Hotel Front-Desk Virtual Assistant

## 1. Overview
The Hotel Front-Desk Virtual Assistant is a conversational AI agent designed to simulate the role of a human front-desk receptionist at a hotel. It assists guests 24/7 through text or voice-based interaction via the hotel's website, mobile app, or in-room smart devices.

---

## 2. Purpose
To automate and streamline common hotel front-desk interactions, reducing wait times, improving guest satisfaction, and minimizing the workload of human staff.

---

## 3. Target Users
- **New Guests** – checking in for the first time
- **Existing Guests** – currently staying at the hotel
- **Prospective Guests** – inquiring about availability and services
- **Departing Guests** – checking out or requesting bills

---

## 4. Core Use Cases

### UC-01: Room Reservation
- **Actor:** Prospective / Returning Guest
- **Goal:** Book a room for specific dates
- **Precondition:** Guest provides travel dates and preferences
- **Main Flow:**
  1. Guest requests a room booking
  2. Assistant asks for check-in/check-out dates
  3. Assistant asks for room type preference (Single, Double, Suite)
  4. Assistant confirms availability
  5. Guest provides personal details and payment info
  6. Assistant confirms booking and sends confirmation number
- **Alternate Flow:** Room not available → Assistant suggests alternative dates or room types

---

### UC-02: Check-In Assistance
- **Actor:** Arriving Guest
- **Goal:** Complete check-in process
- **Precondition:** Guest has an existing reservation
- **Main Flow:**
  1. Guest provides reservation ID or name
  2. Assistant retrieves booking details
  3. Assistant confirms room number and amenities
  4. Assistant provides key card / access instructions
  5. Assistant offers information about hotel services
- **Alternate Flow:** Reservation not found → Escalate to human staff

---

### UC-03: Check-Out Assistance
- **Actor:** Departing Guest
- **Goal:** Complete check-out and receive bill
- **Precondition:** Guest is currently checked in
- **Main Flow:**
  1. Guest requests check-out
  2. Assistant retrieves stay summary and charges
  3. Guest reviews the bill
  4. Assistant processes payment
  5. Assistant thanks guest and requests feedback
- **Alternate Flow:** Billing dispute → Escalate to human staff

---

### UC-04: Room Service Request
- **Actor:** In-House Guest
- **Goal:** Order food, amenities, or housekeeping
- **Main Flow:**
  1. Guest states the type of service needed
  2. Assistant confirms details (items, quantity, timing)
  3. Assistant places the request with the relevant department
  4. Assistant provides an estimated arrival time
- **Alternate Flow:** Service unavailable → Suggest alternatives

---

### UC-05: FAQs & Hotel Information
- **Actor:** Any Guest
- **Goal:** Get information about hotel facilities, policies, location
- **Examples:**
  - Pool/gym timings
  - Wi-Fi password
  - Parking availability
  - Pet policy
  - Nearby attractions

---

### UC-06: Complaint & Escalation Handling
- **Actor:** Any Guest
- **Goal:** Report an issue or complaint
- **Main Flow:**
  1. Guest describes the issue
  2. Assistant logs the complaint
  3. Assistant provides immediate resolution if possible
  4. If unresolved → Escalate to human staff with full context
- **Alternate Flow:** Guest insists on speaking to a human → Immediate escalation

---

## 5. System Boundaries

| In Scope                        | Out of Scope                     |
|---------------------------------|----------------------------------|
| Reservations, Check-in/out      | Financial fraud detection        |
| Room service requests           | Complex legal disputes           |
| FAQ handling                    | Medical emergencies (redirect only) |
| Complaint logging               | Third-party travel bookings      |

---

## 6. Key Assumptions
- The assistant has access to the hotel's Property Management System (PMS)
- Payment processing is handled via a secure integrated gateway
- Human handoff is always available as a fallback
- The assistant supports English as the primary language

---

## 7. Key Benefits
- 24/7 availability without human staffing costs
- Faster response times for common queries
- Consistent and professional guest experience
- Reduced front-desk congestion
- Data collection for improving guest services
