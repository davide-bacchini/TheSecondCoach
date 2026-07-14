# The Second Coach

Built at Hacklytics 2026 (Georgia Tech, Atlanta, Feb 20-22).

## What it does

Every football match has around 3,500 on ball events. Coaches can realistically review a handful. Current analytics measure outcomes: whether a pass was completed, whether a shot was on target. But a good decision that fails gets penalized, and a bad decision that succeeds gets rewarded. There is no separation between the quality of the decision and the quality of the result.

This project grades the decision itself, not the outcome.

Using StatsBomb 360 tracking data, the system freezes time at every action in a match, looks at where every player is positioned, generates the set of alternatives the player could have taken (pass elsewhere, carry, shoot), scores each option based on the threat it creates, and compares what the player actually did against what was available.

## How it works

1. Freeze the moment. Capture the full pitch state (all player positions) at the instant of each action.
2. Generate alternatives. For each event, compute the other actions the player could have taken given the positions around them.
3. Score every option. A model predicts the expected threat value of each possible action.
4. Compare. Was the actual play the best available option, or was there a better path?

The output tells coaches and players where decisions cost the team, what the better option was, and how often specific patterns repeat across a match or a season.

## What coaches and players get

Players see where their decision making costs the team and why. Coaches get feedback grounded in data rather than impression. The system can surface recurring decision habits across full matches or seasons.

For example: "You chose Option A, but Option B was available and carried 4x higher goal probability."

## Data

Built with StatsBomb Open Data (FIFA World Cup 2022 matches).

## Built at

Hacklytics 2026, Georgia Tech, Atlanta. Feb 20-22, 2026.
