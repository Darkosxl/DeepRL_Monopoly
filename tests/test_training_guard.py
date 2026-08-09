import unittest
from unittest.mock import patch

from training_guard import GIB, MemoryLimitReached, MemoryWatchdog


class MemoryWatchdogTests(unittest.TestCase):
    def setUp(self):
        self.watchdog = MemoryWatchdog(
            stop_rss_gib=3,
            hard_rss_gib=4,
            min_available_gib=2,
        )

    def test_check_records_peak_memory(self):
        with (
            patch.object(self.watchdog, "rss_bytes", return_value=2 * GIB),
            patch.object(self.watchdog, "available_bytes", return_value=5 * GIB),
        ):
            rss, available = self.watchdog.check()

        self.assertEqual(rss, 2 * GIB)
        self.assertEqual(available, 5 * GIB)
        self.assertEqual(self.watchdog.peak_rss, 2 * GIB)

    def test_check_stops_at_process_limit(self):
        with (
            patch.object(self.watchdog, "rss_bytes", return_value=3 * GIB),
            patch.object(self.watchdog, "available_bytes", return_value=5 * GIB),
        ):
            with self.assertRaisesRegex(MemoryLimitReached, "checkpoint limit"):
                self.watchdog.check()

    def test_check_stops_when_system_memory_is_low(self):
        with (
            patch.object(self.watchdog, "rss_bytes", return_value=GIB),
            patch.object(self.watchdog, "available_bytes", return_value=2 * GIB),
        ):
            with self.assertRaisesRegex(MemoryLimitReached, "available RAM"):
                self.watchdog.check()

    def test_check_stops_if_memory_cannot_be_measured(self):
        with patch.object(
            self.watchdog,
            "rss_bytes",
            side_effect=MemoryLimitReached("Process RSS is unavailable"),
        ):
            with self.assertRaisesRegex(MemoryLimitReached, "RSS is unavailable"):
                self.watchdog.check()


if __name__ == "__main__":
    unittest.main()
