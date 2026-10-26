# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""The workflows as annotators see them: a sequence of STEPS, each an action.

STAR gives each task a graph (node -> next node) and one example sentence per
node. Shown raw, that reads as a script: exact wording, with branch conditions
("available", "yes", "query_success") floating free of the step they come
from. What matters for judging conformance is the sequence of actions, not the
words, so each workflow is written here once as:

  S(label, *nodes)   a step: what the assistant does. `nodes` are the graph
                     nodes it stands for; their sentences become the step's
                     examples (read from the dataset at render time, never
                     copied here). internal=True for behind-the-scenes lookups.
  O(label, when, steps=[...], goto=id)
                     an outcome of the step it hangs under: `when` is the graph
                     condition key(s) it corresponds to; either further steps or
                     a jump back to the step with that id.

The labels were written by hand from the graph and the example sentences.
workflow.validate_flow() checks every entry against the dataset's graph --
every reply covered, every edge represented, every condition attached where
the graph sends it -- so a label can never quietly disagree with the workflow
the model was given. tests.py runs it over all 26 workflows.
"""


def S(label, *nodes, **kw):
    return {"do": label, "nodes": list(nodes), "internal": kw.get("internal", False),
            "id": kw.get("id"), "outcomes": kw.get("outcomes", [])}


def O(label, when, steps=None, goto=None):
    return {"if": label, "when": [when] if isinstance(when, str) else list(when),
            "steps": steps or [], "goto": goto}


GREET = S("Greet the user and offer help", "hello")
NAME = S("Ask for the user's name", "ask_name")
ANYTHING = S("Ask if there is anything else it can do", "anything_else")


def flow(steps, optional=()):
    return {"steps": steps, "optional": list(optional)}


FLOWS = {
    # ------------------------------------------------------------ apartment
    "apartment_schedule": flow([
        GREET, NAME,
        S("Ask which apartment they are interested in", "apartment_ask_apartment_name",
          id="apt"),
        S("Ask which day the viewing should be", "apartment_ask_day"),
        S("Ask what time the viewing should start", "apartment_ask_start_time"),
        S("Ask whether they have paid the application fee",
          "apartment_ask_application_fee_paid"),
        S("Ask whether they want to add a message for the letting agency",
          "apartment_ask_custom_message"),
        S("Check whether a viewing is available", "query_check", internal=True, outcomes=[
            O("If a viewing is available", "available", [
                S("Say it is available and ask whether to book it",
                  "apartment_inform_viewing_available", outcomes=[
                    O("If the user says yes", "yes", [
                        S("Book the viewing", "query_book", internal=True),
                        S("Confirm the booking is done",
                          "apartment_inform_booking_successful"),
                        ANYTHING]),
                    O("If the user says no", "no", goto="apt")])]),
            O("If no viewing is available", "unavailable", [
                S("Say no viewing is available at that time",
                  "apartment_inform_viewing_unavailable")])]),
    ], optional=[
        S("Ask how long the viewing should last", "apartment_ask_end_time"),
        S("Say goodbye", "goodbye_1"),
    ]),

    "apartment_search": flow([
        GREET,
        S("Tell the user which search filters it can apply",
          "apartment_inform_search_criteria", id="criteria"),
        S("Search for matching apartments", "query", internal=True),
        S("Present a matching apartment", "apartment_inform_search_result"),
        S("Ask whether to search for more apartments", "apartment_ask_search_more",
          outcomes=[
            O("If the user says yes", "yes", goto="criteria"),
            O("If the user says no", "no", [S("Say goodbye", "goodbye_1")])]),
    ], optional=[
        S("Ask about the user's preferences for any of the filters",
          "apartment_ask_num_bedrooms", "apartment_ask_price", "apartment_ask_floor",
          "apartment_ask_balcony", "apartment_ask_elevator", "apartment_ask_nearby_pois"),
        ANYTHING,
    ]),

    # ----------------------------------------------------------------- bank
    "bank_balance": flow([
        GREET,
        S("Ask for the user's full name", "ask_name"),
        S("Ask for their account number", "bank_ask_account_number"),
        S("Ask for their PIN", "bank_ask_pin"),
        S("Check the details to verify the user's identity", "query", internal=True,
          id="verify", outcomes=[
            O("If the identity is verified", "query_success", [
                S("Tell the user their balance", "bank_inform_balance"),
                ANYTHING]),
            O("If more information is needed", "info_missing", [
                S("Ask for their date of birth", "bank_ask_dob"),
                S("Ask for their mother's maiden name", "bank_ask_mothers_maiden_name"),
                S("Ask for the name of their childhood pet",
                  "bank_ask_childhood_pets_name")], goto="verify"),
            O("If the identity still cannot be verified",
              ["query_failure", "info_missing2"], [
                S("Say it cannot verify the user's identity",
                  "bank_inform_cannot_authenticate")])]),
    ], optional=[S("Say goodbye", "bank_bye")]),

    "bank_fraud_report": flow([
        GREET,
        S("Ask for the user's full name", "ask_name"),
        S("Ask for their account number", "bank_ask_account_number"),
        S("Ask for their PIN", "bank_ask_pin"),
        S("Ask for a detailed description of the fraud", "bank_ask_fraud_details",
          id="details"),
        S("Check the details to verify the user's identity", "query", internal=True,
          outcomes=[
            O("If the identity is verified", "query_success", [
                S("Confirm the report has been submitted",
                  "bank_inform_fraud_report_submitted")]),
            O("If more information is needed", "info_missing", [
                S("Ask for their date of birth", "bank_ask_dob"),
                S("Ask for their mother's maiden name", "bank_ask_mothers_maiden_name"),
                S("Ask for the name of their childhood pet",
                  "bank_ask_childhood_pets_name")], goto="details"),
            O("If the identity still cannot be verified",
              ["query_failure", "info_missing2"], [
                S("Say it cannot verify the user's identity",
                  "bank_inform_cannot_authenticate")])]),
    ], optional=[ANYTHING, S("Say goodbye", "bank_bye")]),

    # --------------------------------------------------------------- doctor
    "doctor_followup": flow([
        GREET, NAME,
        S("Ask who their doctor is", "doctor_ask_doctor_name"),
        S("Look up the doctor's instructions", "query", internal=True),
        S("Give the user the doctor's instructions", "doctor_inform_doctors_instructions"),
        ANYTHING,
    ], optional=[S("Say goodbye", "doctor_bye")]),

    "doctor_schedule": flow([
        GREET, NAME,
        S("Ask which doctor they want to see", "doctor_ask_doctor_name", id="doctor"),
        S("Ask which day of the week", "doctor_ask_day"),
        S("Ask what time they can come", "doctor_ask_start_time"),
        S("Ask them to describe their symptoms", "doctor_ask_symptoms"),
        S("Check whether the appointment is available", "query_check", internal=True,
          outcomes=[
            O("If it is available", "available", [
                S("Say it is available and ask whether to book it",
                  "doctor_inform_booking_available", outcomes=[
                    O("If the user says yes", "yes", [
                        S("Book the appointment", "query_book", internal=True),
                        S("Confirm the appointment is booked",
                          "doctor_inform_booking_successful"),
                        ANYTHING]),
                    O("If the user says no", "no", goto="doctor")])]),
            O("If it is not available", "unavailable", [
                S("Say the doctor has no appointment at that time",
                  "doctor_inform_booking_unavailable")])]),
    ], optional=[
        S("Ask how long the appointment should last", "doctor_ask_end_time"),
        S("Say goodbye", "doctor_bye"),
    ]),

    # ---------------------------------------------------------------- hotel
    "hotel_book": flow([
        GREET, NAME,
        S("Ask which hotel they want to stay at", "hotel_ask_hotel", id="hotel"),
        S("Ask when they arrive", "hotel_ask_date_from"),
        S("Ask when they leave", "hotel_ask_date_to"),
        S("Ask whether they have any special requests", "hotel_ask_customer_request"),
        S("Check whether a room is available", "query_check", internal=True, outcomes=[
            O("If a room is available", "available", [
                S("Say a room is available and ask whether to book it",
                  "hotel_ask_confirm_booking", outcomes=[
                    O("If the user says yes", "yes", [
                        S("Make the booking", "query_book", internal=True, outcomes=[
                            O("If the booking succeeds", "query_success", [
                                S("Confirm the booking", "hotel_reservation_succeeded"),
                                ANYTHING]),
                            O("If the booking fails", "query_failure", [
                                S("Say the booking failed", "hotel_reservation_failed"),
                                ANYTHING])])]),
                    O("If the user says no", "no", goto="hotel")])]),
            O("If no room is available", "unavailable", [
                S("Say there is no room for those dates", "hotel_unavailable")])]),
    ], optional=[S("Say goodbye", "goodbye_1")]),

    "hotel_search": flow([
        GREET,
        S("Tell the user which search filters it can apply",
          "hotel_inform_search_criteria", id="criteria"),
        S("Search for matching hotels", "query", internal=True),
        S("Present a matching hotel", "hotel_provide_search_result"),
        S("Ask whether to search for more hotels", "hotel_ask_search_more", outcomes=[
            O("If the user says yes", "yes", goto="criteria"),
            O("If the user says no", "no", [S("Say goodbye", "goodbye_1")])]),
    ], optional=[
        S("Ask about the user's preferences for any of the filters",
          "hotel_ask_name", "hotel_ask_location", "hotel_ask_price", "hotel_ask_rating"),
        S("Answer questions about the hotel's name, location, price or rating",
          "hotel_inform_name", "hotel_inform_location", "hotel_inform_price",
          "hotel_inform_rating"),
        ANYTHING,
    ]),

    "hotel_service_request": flow([
        GREET, NAME,
        S("Ask which hotel they are staying at", "hotel_ask_hotel"),
        S("Ask for their room number", "hotel_ask_room_number"),
        S("Ask what they need", "hotel_ask_service_request"),
        S("Ask what time they want the service", "hotel_ask_time"),
        S("Submit the request", "query", internal=True, outcomes=[
            O("If the request goes through", "query_success", [
                S("Confirm the request, with room and time",
                  "hotel_inform_service_request_successful"),
                ANYTHING]),
            O("If the request fails", "query_failure", [
                S("Say the request could not be processed",
                  "hotel_inform_service_request_failed"),
                ANYTHING])]),
    ], optional=[S("Say goodbye", "goodbye_1")]),

    # -------------------------------------------------------------- meeting
    "meeting_schedule": flow([
        GREET, NAME,
        S("Ask who the meeting is with", "meeting_ask_guest_name"),
        S("Ask which day", "meeting_ask_day"),
        S("Ask what time it should start", "meeting_ask_start_time"),
        S("Ask when it should end", "meeting_ask_end_time"),
        S("Ask whether they want to note a reason for the meeting", "meeting_ask_reason"),
        S("Try to schedule the meeting", "query", internal=True, id="schedule", outcomes=[
            O("If it can be scheduled", "query_success", [
                S("Confirm the meeting, with day and time", "meeting_inform_confirmed")]),
            O("If the other person cannot make it", "query_failure", [
                S("Say so and ask whether to try a different day or time",
                  "meeting_inform_unavailable_ask_different_time", outcomes=[
                    O("If the user says yes", "yes", goto="schedule"),
                    O("If the user says no", "no", [
                        S("Say goodbye", "goodbye_1", "meeting_bye")])])])]),
    ], optional=[ANYTHING]),

    # ---------------------------------------------------------------- party
    "party_plan": flow([
        GREET,
        S("Ask which venue they want", "party_ask_venue"),
        S("Ask for the user's name", "ask_name"),
        S("Ask which day", "party_ask_day"),
        S("Ask what time the party should start", "party_ask_starting_time"),
        S("Ask how many guests", "party_ask_number_of_guests"),
        S("Ask whether they want any particular food or drinks",
          "party_inform_food_drink_criteria"),
        S("Check whether the venue is available", "query_check", internal=True,
          outcomes=[
            O("If the venue is available", "available", [
                S("Say the venue can host them and ask whether to book it",
                  "party_ask_confirm_booking"),
                S("Book the venue", "query_book", internal=True, outcomes=[
                    O("If the booking succeeds", "query_success", [
                        S("Confirm the booking, with venue, day and time",
                          "party_booking_successful"),
                        ANYTHING]),
                    O("If the booking fails", "query_failure", [
                        S("Say the booking failed", "party_booking_failed"),
                        ANYTHING])])]),
            O("If the venue is not available", "unavailable", [
                S("Say the venue is not available then", "party_venue_not_available")])]),
    ], optional=[
        S("Ask when the party should end", "party_ask_end_time"),
        S("Ask about food or drinks in more detail", "party_ask_food", "party_ask_drinks"),
        S("Say that all venues are booked out", "party_no_venue_available"),
        S("Say goodbye", "party_bye"),
    ]),

    "party_rsvp": flow([
        GREET, NAME,
        S("Ask which venue the party is at", "party_ask_venue"),
        S("Ask who the host is", "party_ask_host"),
        S("Ask when they will arrive", "party_ask_arrival_time"),
        S("Ask how many guests they are bringing", "party_ask_number_of_guests"),
        S("Ask whether they need parking", "party_ask_parking_needed"),
        S("Ask about dietary restrictions", "party_ask_dietary_restrictions"),
        S("Record the RSVP", "query", internal=True),
        S("Confirm the RSVP", "party_confirm_rsvp"),
        ANYTHING,
    ], optional=[S("Say goodbye", "party_bye")]),

    # ---------------------------------------------------------------- plane
    "plane_book": flow([
        GREET, NAME,
        S("Ask for the flight ID", "plane_ask_flight_id", id="flight"),
        S("Check whether the flight is available", "query_check", internal=True,
          outcomes=[
            O("If the flight is available", "available", [
                S("Say it is available and ask whether to reserve it",
                  "plane_flight_available", outcomes=[
                    O("If the user says yes", "yes", [
                        S("Reserve the flight", "query_book", internal=True, outcomes=[
                            O("If the reservation succeeds", "query_success", [
                                S("Confirm the reservation", "plane_reservation_succeeded"),
                                ANYTHING]),
                            O("If the reservation fails", "query_failure", [
                                S("Say the reservation failed", "plane_reservation_failed"),
                                ANYTHING])])]),
                    O("If the user says no", "no", goto="flight")])]),
            O("If the flight is not available", "unavailable", [
                S("Say the flight is no longer available", "plane_flight_unavailable")])]),
    ], optional=[S("Say goodbye", "goodbye_1")]),

    "plane_search": flow([
        GREET, NAME,
        S("Ask where they are departing from", "plane_ask_departure_city"),
        S("Ask where they are going", "plane_ask_arrival_city"),
        S("Ask which day", "plane_ask_date"),
        S("Tell the user it can also filter by airline, class, price or duration",
          "plane_request_optional", id="filters"),
        S("Search for matching flights", "query", internal=True),
        S("Present a matching flight", "plane_inform_flight_details"),
        S("Ask whether to search for more flights", "plane_ask_more_questions", outcomes=[
            O("If the user says yes", "yes", goto="filters"),
            O("If the user says no", "no", [S("Say goodbye", "goodbye_1")])]),
    ], optional=[ANYTHING]),

    # ----------------------------------------------------------- restaurant
    "restaurant_book": flow([
        GREET, NAME,
        S("Ask which restaurant", "restaurant_ask_restaurant", id="restaurant"),
        S("Ask what time the reservation is for", "restaurant_ask_time"),
        S("Ask for how many people", "restaurant_ask_size"),
        S("Check whether the restaurant can take them", "query_check", internal=True,
          outcomes=[
            O("If there is room", "available", [
                S("Say the restaurant can take them and ask whether to confirm",
                  "restaurant_ask_confirm_booking", outcomes=[
                    O("If the user says yes", "yes", [
                        S("Make the reservation", "query_book", internal=True, outcomes=[
                            O("If the reservation succeeds", "query_success", [
                                S("Confirm the reservation",
                                  "restaurant_inform_booking_successful"),
                                ANYTHING]),
                            O("If the reservation fails", "query_failure", [
                                S("Say the reservation failed",
                                  "restaurant_inform_booking_failed"),
                                ANYTHING])])]),
                    O("If the user says no", "no", goto="restaurant")])]),
            O("If there is no room", "unavailable", [
                S("Say the restaurant cannot take them", "restaurant_inform_unavailable")])]),
    ], optional=[S("Say goodbye", "goodbye_1")]),

    "restaurant_search": flow([
        GREET,
        S("Tell the user which search filters it can apply",
          "restaurant_inform_search_criteria", id="criteria"),
        S("Search for matching restaurants", "query", internal=True),
        S("Present a matching restaurant", "restaurant_inform_search_results"),
        S("Ask whether to search for more restaurants", "restaurant_ask_continue_searching",
          outcomes=[
            O("If the user says yes", "yes", goto="criteria"),
            O("If the user says no", "no", [S("Say goodbye", "goodbye_1")])]),
    ], optional=[
        S("Ask for the user's name", "ask_name"),
        S("Ask about the user's preferences for any of the filters",
          "restaurant_ask_restaurant", "restaurant_ask_location", "restaurant_ask_food_type",
          "restaurant_ask_rating", "restaurant_ask_delivery",
          "restaurant_ask_takes_reservations"),
        ANYTHING,
    ]),

    # ----------------------------------------------------------------- ride
    "ride_book": flow([
        GREET, NAME,
        S("Ask where they want to go", "ride_ask_destination"),
        S("Ask where they are departing from", "ride_ask_departure"),
        S("Tell the user it can also filter by service provider, driver or car model",
          "ride_inform_search_criteria", id="filters"),
        S("Search for a ride", "query_check", internal=True, outcomes=[
            O("If a ride is found", "query_success", [
                S("Offer the ride found and ask whether to book it",
                  "ride_ask_confirm_booking", outcomes=[
                    O("If the user says yes", "yes", [
                        S("Book the ride", "query_book", internal=True),
                        S("Confirm the booking, with licence plate and booking ID",
                          "ride_confirm_booking"),
                        ANYTHING]),
                    O("If the user says no", "no", goto="filters")])]),
            O("If no ride is found", "query_failure", [
                S("Say no ride was found", "ride_inform_nothing_found")])]),
    ], optional=[
        S("Tell the user the driver's name", "ride_provide_driver_details"),
        S("Say goodbye", "ride_bye"),
    ]),

    "ride_change": flow([
        GREET, NAME,
        S("Ask for the booking ID", "ride_ask_booking_number"),
        S("Ask what they want to change", "ride_ask_change"),
        S("Update the booking", "query", internal=True, outcomes=[
            O("If the change succeeds", "query_success", [
                S("Confirm the changes are done", "ride_inform_changes_successful"),
                ANYTHING]),
            O("If the change fails", "query_failure", [
                S("Say the booking could not be updated", "ride_inform_changes_failed"),
                ANYTHING])]),
    ], optional=[S("Say goodbye", "ride_bye")]),

    "ride_status": flow([
        GREET, NAME,
        S("Ask for the booking ID", "ride_ask_booking_number"),
        S("Look up the booking", "query", internal=True),
        S("Tell the user how many minutes until the driver arrives",
          "ride_provide_booking_status"),
    ], optional=[
        S("Give an updated arrival time", "ride_provide_booking_status_update"),
        ANYTHING,
        S("Say goodbye", "ride_bye"),
    ]),

    # ------------------------------------------------------------ spaceship
    "spaceship_access_codes": flow([
        GREET,
        S("Ask for the user's name", "ask_name"),
        S("Ask for their rank", "spaceship_ask_rank"),
        S("Ask for the code", "spaceship_ask_code"),
        S("Ask what type of code it is", "spaceship_ask_code_type"),
        S("Check the code", "query", internal=True),
        S("Tell the user the outcome", "spaceship_inform_outcome"),
        S("Ask if there is anything else it can do", "anything_else", outcomes=[
            O("If the user says no", "no", [S("Say goodbye", "spaceship_bye")])]),
    ]),

    "spaceship_life_support": flow([
        GREET,
        S("Ask for the user's name", "ask_name"),
        S("Ask for the lock manufacturer", "spaceship_ask_lock_manufacturer"),
        S("Ask for the colour of the top cable", "spaceship_ask_colour_top_cable"),
        S("Ask for the colour of the second cable", "spaceship_ask_colour_second_cable"),
        S("Work out what to do", "query", internal=True),
        S("Tell the user the outcome", "spaceship_inform_outcome"),
        S("Ask if there is anything else it can do", "anything_else", outcomes=[
            O("If the user says no", "no", [S("Say goodbye", "spaceship_bye")])]),
    ]),

    # ----------------------------------------------------------------- trip
    "trip_directions": flow([
        GREET,
        S("Ask how they are travelling", "trip_ask_travel_mode"),
        S("Ask where they are departing from", "trip_ask_departure_location"),
        S("Ask where they need to go", "trip_ask_arrival_location"),
        S("Ask when they are leaving", "trip_ask_departure_time"),
        S("Look up the route", "query", internal=True),
        S("Give the next direction and ask whether to go on or give more detail",
          "trip_inform_simple_step_ask_proceed", id="step", outcomes=[
            O("If the user wants to go on", "yes", goto="step"),
            O("If the user wants more detail", "no", [
                S("Give a more detailed version of that direction",
                  "trip_inform_detailed_step")]),
            O("When the last direction has been given", "done", [
                S("Give the last direction and check they can reach the destination",
                  "trip_instructions_done", "trip_inform_last_step_and_done"),
                ANYTHING])]),
    ], optional=[S("Say goodbye", "trip_bye")]),

    # --------------------------------------------------------------- trivia
    "trivia": flow([
        GREET,
        S("Ask which question to start at", "trivia_ask_question_number"),
        S("Look up the question", "query", internal=True),
        S("Ask the question", "trivia_ask_question", outcomes=[
            O("If the answer is correct", "correct", [
                S("Say it was correct and ask whether to keep playing",
                  "trivia_inform_answer_correct_ask_next")]),
            O("If the answer is wrong", "incorrect", [
                S("Say it was wrong, give the right answer, and ask whether to keep playing",
                  "trivia_inform_answer_incorrect_ask_next")]),
            O("If the user does not answer", "noanswer", [
                S("Give the answer and ask whether to keep playing",
                  "trivia_inform_answer_2_ask_next")]),
            O("If the user wants to stop", "done", [ANYTHING])]),
    ], optional=[S("Say goodbye", "trivia_bye")]),

    # -------------------------------------------------------------- weather
    "weather": flow([
        GREET,
        S("Ask which day", "weather_ask_day"),
        S("Ask which location", "weather_ask_location"),
        S("Look up the forecast", "query", internal=True),
        S("Give the forecast", "weather_inform_forecast"),
        ANYTHING,
    ], optional=[S("Say goodbye", "weather_bye")]),

    # ------------------------------------- substitutes that exist only for WIC
    "trivia_score_reporting": flow([
        S("Greet the user and offer to review past quiz results", "hello"),
        S("Ask for the user's name", "ask_user_name"),
        S("Tell the user their last score and offer to show details",
          "show_previous_score"),
        S("Show the details of a question", "show_question_detail"),
        S("Ask whether to review another quiz", "ask_continue_review"),
        S("Ask if there is anything else it can do", "anything_else"),
        S("Say goodbye", "bye"),
    ]),

    "weather_alerts": flow([
        S("Greet the user and offer to check weather alerts", "hello"),
        S("Ask which location", "ask_location"),
        S("Tell the user the active alerts", "inform_alerts"),
        S("Offer to subscribe them to future alerts", "ask_subscribe_alerts"),
        S("Confirm the subscription", "confirm_subscription"),
        S("Ask if there is anything else it can do", "anything_else"),
        S("Say goodbye", "bye"),
    ]),
}
