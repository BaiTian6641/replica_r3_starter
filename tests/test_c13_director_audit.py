import json
from replica_lab.runtime.director import DirectorBudget,LearningDirector,OutcomeSummary
def test_success_no_update_and_large_action_proposal():
 d=LearningDirector(budget=DirectorBudget(max_cost_units=10000))
 succ=d.review([OutcomeSummary('f','procedure',3,3,('e1','e2'),role='development',verified_receipts=3,cost_units=1.0)])
 assert succ.status=='no_update'
 fail=d.review([OutcomeSummary('f','action',9,0,('e1','e2'),role='development',verified_receipts=0,cost_units=1.0)])
 assert fail.status=='proposed' and fail.proposal.target_kind=='neural'
