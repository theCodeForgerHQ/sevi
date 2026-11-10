# Sevi (செவி): a hearing screen in Tamil and English

Great Indian AI Internship Challenge. Team of two: Nithisha Venkatesh and Ajayaditya Lokchandra.

## The problem

Hearing loss comes on slowly. Most people don't get tested until they are already asking others to repeat themselves, and a test usually means a trip to a clinic. The hearing tests you can take online are mostly in English and are answered by tapping buttons. If you are a Tamil speaker, and especially an older one, there isn't much you can take at home in your own language that tells you whether you should see a doctor.

Sevi is for those people, and for the son or granddaughter who sets it up for them. You need a browser and a pair of earphones.

## How it works

Sevi uses a digits-in-noise test. This kind of test is already used in hearing research. You hear three digits spoken over background noise and say them out loud. If you get them right, the next round has more noise. If you miss, it has less. After one practice round and 24 scored rounds, the test has found the noise level where you start losing speech. That number is your result, in dB SNR.

You get the result in Tamil or English with a short explanation of what it means. If it is worth getting checked, Sevi says so and points you to an audiologist or an ENT doctor. It is a screening, not a diagnosis, and it never tells anyone their hearing is fine.

## Where Gnani's models come in

Your spoken answers stream to Gnani Prisma, which turns them into digits. We built the test around voice because Prisma handled Tamil digits well in our testing. A tap-only version would have been easier to build but harder for the people we built it for.

Everything you hear in the app is Gnani Timbre: the instructions, the consent text and the digits themselves, in both languages. We generate each clip once and store it, so taking a test doesn't spend credits.

The explanation of your result is written by Evon, which we run ourselves from the open weights. Before you see it, a guard checks the text for things we never want to say (like "your hearing is normal") and for length. If Evon is down, Sevi shows wording we wrote and reviewed by hand.

## How it's built

The frontend is a plain browser app and the backend is in Python. The page, the API and the live audio connection all come from the same server. The noise level for each round comes from an adaptive staircase. Each session has a fixed credit budget, so one user can't run up the bill.

All the text users see sits in one file, and the tests check it: no banned phrases, "not a diagnosis" where it needs to be, and consent text that names who handles your data and how long they keep it. A native speaker reviewed every Tamil line. The code is open source at github.com/theCodeForgerHQ/sevi.

## Who did what

Nithisha Venkatesh wrote the user-facing text, did the Tamil translation and review, wrote the consent and safety wording, and wrote the explainer's prompts and checks.

Ajayaditya Lokchandra built the web app and backend, connected Prisma and Timbre, wrote the adaptive test logic, set up Evon hosting, and handled deployment.

We split the pull requests evenly, and each of us reviewed and merged the other's work.

## Next

We want to add more Indian languages, since Prisma and Timbre already support more than ten. After that, a version you can take over a normal phone call, for people without a smartphone.
