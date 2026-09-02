## ADDED Requirements

### Requirement: Deterministic final-answer extraction
The evaluator SHALL extract final answers from `<answer>...</answer>` or `Final Answer:` output conventions and return no answer for malformed content.

#### Scenario: Tagged answer
- **WHEN** output contains `<answer>42</answer>`
- **THEN** the extracted answer is `42`

### Requirement: Separable chart rewards
The evaluator SHALL report correctness and format reward components independently and aggregate accuracy and format rate.

#### Scenario: Correct malformed answer
- **WHEN** output has the correct value but lacks the configured final-answer format
- **THEN** correctness is positive and format reward is zero
