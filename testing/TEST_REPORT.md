# EmpathyCo Test Report

## Summary
- Total test cases: 17 + 2 multilingual + 1 transcription

## Detailed Results

| Emotion | Input Type | Input Content | Expected | Predicted | Confidence | Pass/Fail | Screenshot |
|---------|------------|---------------|----------|-----------|------------|-----------|-------------|
| neutral | text | The table is made of wood. | neutral | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_neutral_text.png) |
| neutral | audio | Audio Upload | neutral | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_neutral_audio.png) |
| calm | text | Everything is peaceful and quiet. | calm | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_calm_text.png) |
| calm | audio | Audio Upload | calm | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_calm_audio.png) |
| happy | text | I am so excited and joyful! | happy | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_happy_text.png) |
| happy | audio | Audio Upload | happy | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_happy_audio.png) |
| sad | text | I feel very sad and lonely. | sad | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_sad_text.png) |
| sad | audio | Audio Upload | sad | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_sad_audio.png) |
| angry | text | I am absolutely furious! | angry | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_angry_text.png) |
| angry | audio | Audio Upload | angry | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_angry_audio.png) |
| fearful | text | I am terrified of what might happen. | fearful | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_fearful_text.png) |
| fearful | audio | Audio Upload | fearful | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_fearful_audio.png) |
| disgust | text | This is disgusting and repulsive. | disgust | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_disgust_text.png) |
| disgust | audio | Audio Upload | disgust | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_disgust_audio.png) |
| surprised | text | Wow! That is completely unexpected! | surprised | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_surprised_text.png) |
| surprised | audio | Audio Upload | surprised | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_surprised_audio.png) |
| Fusion | text+audio | Happy Text + Angry Audio | fusion | CHECK_IMG | CHECK_IMG | Done | [screenshot](screenshots/test_fusion_happy_angry.png) |

## Multilingual Results

| Language | Text | Expected | Predicted | Confidence | Screenshot |
|----------|------|----------|-----------|------------|------------|
| Hindi | अरे! क्या यह तु... | angry | CHECK_IMG | CHECK_IMG | [screenshot](screenshots/test_angry_hi_multilingual.png) |
| Marathi | मी खूप आनंदात आ... | happy | CHECK_IMG | CHECK_IMG | [screenshot](screenshots/test_happy_mr_multilingual.png) |

## Auto-transcribe Test
- Input audio: RAVDESS sample (Happy)
- Transcribed text: (See screenshot screenshots/test_auto_transcribe.png)
- Predicted emotion: (See screenshot)
