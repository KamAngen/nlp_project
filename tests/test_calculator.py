from legal_agent.agent.tools import SafeCalculator


def test_safe_calculator_basic_expression():
    calculator = SafeCalculator()
    result = calculator.evaluate("(20000 - 5000 - 3000 - 2000) * 0.1 - 210")
    assert result == 790.0


def test_safe_calculator_supports_round_and_abs():
    calculator = SafeCalculator()
    result = calculator.evaluate("round(abs(-12.345), 2)")
    assert result == 12.35


def test_safe_calculator_supports_percent_and_chinese_units():
    calculator = SafeCalculator()
    result = calculator.evaluate("(132万 * 38%) * 25% * 1.5")
    assert result == 188100.0
