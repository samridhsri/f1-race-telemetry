"""
Quick test to check if FastF1 API has 2025 F1 race data available
Run this in the Streamlit container: docker exec streamlit python /app/test_fastf1_2025.py
"""
import fastf1
from datetime import datetime

print("=" * 70)
print("FastF1 2025 Data Availability Check")
print("=" * 70)
print(f"FastF1 version: {fastf1.__version__}")
print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}\n")

# Test 1: Get 2025 schedule
print("[1] Fetching 2025 F1 Race Schedule...")
try:
    schedule = fastf1.get_event_schedule(2025)
    print(f"✓ Found {len(schedule)} races in 2025 calendar\n")

    # Show first few races
    print("Sample races:")
    for idx, row in schedule.head(5).iterrows():
        print(f"  Round {row['RoundNumber']}: {row['EventName']} - {row['EventDate']}")

    # Check which races have happened
    current_date = datetime.now()
    completed = 0
    upcoming = 0

    for idx, row in schedule.iterrows():
        event_date = row['EventDate']
        if isinstance(event_date, str):
            event_date = datetime.fromisoformat(event_date.replace('Z', '+00:00')).replace(tzinfo=None)

        if event_date < current_date:
            completed += 1
        else:
            upcoming += 1

    print(f"\n✓ Race Status: {completed} completed, {upcoming} upcoming")

    if completed == 0:
        print("\n⚠ WARNING: No races completed yet in 2025!")
        print("  → The 2025 F1 season hasn't started")
        print("  → Cannot use 2025 data for training yet")
        print("\n📋 RECOMMENDATION:")
        print("  1. Change training years from '2023,2024,2025' to '2023,2024'")
        print("  2. Keep predicting 2026 (that's fine!)")
        print("  3. Add 2025 data later when races complete")
    elif completed < 3:
        print(f"\n⚠ WARNING: Only {completed} race(s) completed")
        print("  → Limited 2025 data available")
        print("  → Model training may not be optimal")
        print("\n📋 RECOMMENDATION:")
        print("  - Consider using only 2023-2024 for now")
        print("  - Wait for more 2025 races before including it")
    else:
        print(f"\n✓ Good! {completed} races completed")
        print("  → You can use 2023,2024,2025 for training")

except Exception as e:
    print(f"✗ ERROR: {e}")
    print("  FastF1 API might be unavailable")

print("\n" + "=" * 70)
