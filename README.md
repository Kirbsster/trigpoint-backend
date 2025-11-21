# trigpoint

## Quick start locally (backend)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit values
uvicorn app.main:app --reload




## github
# init
git init
add .
git commit -m "Initial"
git remote add origin git@github.com:Kirbsster/trigpoint-backend.git
git branch -M main
git push -u origin main
# push change
git add .
git commit -m "comment"
git push
# create dev branch
git checkout main
git pull
git checkout -b feature/auth
git push -u origin feature/auth
#... work and commit stuff
# merge back with main
git checkout main
git pull
git merge feature/auth
git push
# revert the last commit on main
git checkout main
git pull
git revert HEAD
git push



## google cloud run - on cloud shell
cd trigpoint-backend
git pull origin main
gcloud builds submit --tag gcr.io/trigpoint/trigpoint-backend
gcloud run deploy trigpoint-backend \
  --image gcr.io/trigpoint/trigpoint-backend \
  --platform managed \
  --region europe-west2 \
  --allow-unauthenticated