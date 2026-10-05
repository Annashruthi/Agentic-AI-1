import unittest

from resource_sharing_agent import Decision, ResourceClaim, ResourceNegotiator


class ResourceNegotiatorTests(unittest.TestCase):
    def setUp(self) -> None:
        claims = [
            ResourceClaim("alpha", requested=8, minimum=2),
            ResourceClaim("beta", requested=8, minimum=2),
        ]
        self.agent = ResourceNegotiator("alpha", claims)

    def test_counter_offer_concedes_toward_fair_share(self) -> None:
        opening = self.agent.make_counter_offer(10, round_number=0)
        final = self.agent.make_counter_offer(10, round_number=5)

        self.assertEqual(opening["alpha"], 8)
        self.assertEqual(final["alpha"], 5)
        self.assertAlmostEqual(sum(final.values()), 10)

    def test_response_counters_early_and_accepts_at_fair_share(self) -> None:
        proposal = {"alpha": 6, "beta": 4}

        early = self.agent.respond(proposal, 10, round_number=0)
        final = self.agent.respond(proposal, 10, round_number=5)

        self.assertEqual(early.decision, Decision.COUNTER)
        self.assertEqual(final.decision, Decision.ACCEPT)

    def test_rejects_infeasible_proposal(self) -> None:
        response = self.agent.respond({"alpha": 7, "beta": 4}, 10, round_number=2)

        self.assertEqual(response.decision, Decision.REJECT)


if __name__ == "__main__":
    unittest.main()