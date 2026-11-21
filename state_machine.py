# logic/state_machine.py

# --------------------------------------------------
# CONFIG (REALISTIC & TUNABLE)
# --------------------------------------------------

WARNING_TIME = 5.0              # seconds → ATTENDED → POTENTIALLY_UNATTENDED
UNATTENDED_CONFIRM_TIME = 7.0   # seconds → POTENTIALLY_UNATTENDED → UNATTENDED
# TOTAL = 5 + 7 = 12 seconds


def update_state(obj, person_confirmed_absent, is_stationary, now, min_state_time):
    """
    Stable, human-realistic state transitions.

    States:
        ATTENDED
        POTENTIALLY_UNATTENDED
        UNATTENDED
        RESOLVED
    """

    # ==================================================
    # ATTENDED (0–5 sec)
    # ==================================================
    if obj.state == "ATTENDED":

        # Person gone + object stationary → start warning phase
        if person_confirmed_absent and is_stationary:
            if now - obj.state_enter_time >= WARNING_TIME:
                obj.state = "POTENTIALLY_UNATTENDED"
                obj.state_enter_time = now

    # ==================================================
    # POTENTIALLY_UNATTENDED (5–12 sec)
    # ==================================================
    elif obj.state == "POTENTIALLY_UNATTENDED":

        # Person came back → reset immediately
        if not person_confirmed_absent:
            obj.state = "ATTENDED"
            obj.state_enter_time = now

        # Still absent + stationary → confirm unattended
        elif is_stationary:
            if now - obj.state_enter_time >= UNATTENDED_CONFIRM_TIME:
                obj.state = "UNATTENDED"
                obj.state_enter_time = now

        # Object moved → reset
        else:
            obj.state = "ATTENDED"
            obj.state_enter_time = now

    # ==================================================
    # UNATTENDED (12+ sec)
    # ==================================================
    elif obj.state == "UNATTENDED":
        # Frozen until resolution
        pass