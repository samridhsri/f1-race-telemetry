"""
Test script to check if FastF1 API has 2025 F1 race data available
"""
import fastf1
import sys
from datetime import datetime

print("=" * 70)
print("FastF1 2025 Data Availability Test")
print("=" * 70)
print(f"\nFastF1 version: {fastf1.__version__}")
print(f"Current date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Test 1: Check if 2025 schedule is available
print("\n[TEST 1] Checking 2025 F1 Race Schedule...")
print("-" * 70)
try:
    schedule_2025 = fastf1.get_event_schedule(2025)
    print(f"✓ SUCCESS: Found {len(schedule_2025)} events in 2025 calendar")
    print("\n2025 F1 Events:")
    print(schedule_2025[['RoundNumber', 'EventName', 'EventDate', 'Country']].to_string())
except Exception as e:
    print(f"✗ FAILED: Could not fetch 2025 schedule")
    print(f"Error: {e}")
    sys.exit(1)

# Test 2: Check if any race has been completed in 2025
print("\n\n[TEST 2] Checking for Completed 2025 Races...")
print("-" * 70)

completed_races = []
current_date = datetime.now()

for idx, event in schedule_2025.iterrows():
    event_date = event['EventDate']
    if isinstance(event_date, str):
        event_date = datetime.fromisoformat(event_date.replace('Z', '+00:00')).replace(tzinfo=None)

    if event_date < current_date:
        completed_races.append({
            'round': event['RoundNumber'],
            'name': event['EventName'],
            'date': event_date
        })

if completed_races:
    print(f"✓ Found {len(completed_races)} completed race(s) in 2025:")
    for race in completed_races:
        print(f"  - Round {race['round']}: {race['name']} ({race['date'].strftime('%Y-%m-%d')})")
else:
    print("⚠ No completed races found in 2025 yet")
    print("  → This is normal if the 2025 season hasn't started")
    print("  → You can use 2023-2024 data for training until races are completed")

# Test 3: Try to fetch session data from the first completed race (if any)
if completed_races:
    print("\n\n[TEST 3] Testing Session Data Fetch from Completed Race...")
    print("-" * 70)

    first_race = completed_races[0]
    try:
        print(f"Attempting to load: {first_race['name']} (Round {first_race['round']})")
        session = fastf1.get_session(2025, first_race['round'], 'R')  # 'R' for Race
        print("  → Loading session data (this may take a moment)...")
        session.load()

        print(f"✓ SUCCESS: Session data loaded!")
        print(f"  - Laps recorded: {len(session.laps)}")
        print(f"  - Drivers: {len(session.drivers)}")

        # Check if results are available
        if hasattr(session, 'results') and session.results is not None:
            print(f"  - Race results: Available ({len(session.results)} entries)")
        else:
            print(f"  - Race results: Not available yet")

    except Exception as e:
        print(f"✗ FAILED: Could not load session data")
        print(f"Error: {e}")
        print("\nPossible reasons:")
        print("  1. Race data not yet published by F1 (usually available 1-2 hours after race)")
        print("  2. API server temporarily unavailable")
        print("  3. Network connectivity issue")
else:
    print("\n\n[TEST 3] Skipping Session Data Test")
    print("-" * 70)
    print("⚠ No completed races to test - 2025 season hasn't started yet")

# Test 4: Recommendation
print("\n\n[RECOMMENDATION]")
print("=" * 70)

if len(completed_races) >= 3:
    print("✓ You have enough 2025 race data to include it in training!")
    print(f"  → {len(completed_races)} races completed")
    print("  → Update training years to: 2023,2024,2025")
elif len(completed_races) > 0:
    print("⚠ Limited 2025 data available")
    print(f"  → Only {len(completed_races)} race(s) completed")
    print("  → Consider waiting for more races or use: 2023,2024")
    print("  → You can still predict 2026, just with less 2025 training data")
else:
    print("⚠ No 2025 race data available yet")
    print("  → The 2025 F1 season hasn't started")
    print("  → Current recommendation: Use training years 2023,2024")
    print("  → Update to 2023,2024,2025 once races are completed")
    print("\nFor 2026 predictions with current setup:")
    print("  1. Keep DRIVERS_2026 lineup as configured")
    print("  2. Use 2023-2024 historical data for training")
    print("  3. Add 2025 data incrementally as races complete")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)
