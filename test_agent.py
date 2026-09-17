import unittest
from importlib import import_module
from types import SimpleNamespace


class AnalystGraphTest(unittest.TestCase):
    def test_jev_routes_clarification(self):
        module = import_module("analyst_langgraph.agent")
        original = module._ask_jev

        def fake_jev(_, questions):
            answers = {
                "route": SimpleNamespace(choice="clarify", confidence=1),
                "is_safe": SimpleNamespace(noul=1),
                "should_remember": SimpleNamespace(noul=0),
                "is_complete": SimpleNamespace(noul=1),
                "escalation": SimpleNamespace(choice="continue", confidence=1),
                "output_quality": SimpleNamespace(score=1, confidence=1),
                "memory_action": SimpleNamespace(choice="reject", confidence=1),
            }
            return {name: answers[name] for name in questions}

        module._ask_jev = fake_jev
        try:
            result = module.agent.invoke({"messages": [{"role": "user", "content": "ambiguous"}]})
        finally:
            module._ask_jev = original

        self.assertEqual(result["analyst"]["route"], "clarify")
        self.assertTrue(result["messages"][-1].content.startswith("Could you clarify"))


if __name__ == "__main__":
    unittest.main()
