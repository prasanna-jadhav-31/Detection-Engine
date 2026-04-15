# Deployment

## Backend on Render

Create a new Render Web Service from this repository root.

- Runtime: `Python`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

Set these environment variables in Render:

- `ALLOWED_ORIGINS=https://your-frontend-domain.vercel.app`

If you use a Vercel preview domain pattern, include each required origin as a comma-separated list.

## Frontend on Vercel

Create the Vercel project using `frontend/` as the project root.

- Framework preset: `Vite`
- Build command: `npm run build`
- Output directory: `dist`

Set this environment variable in Vercel:

- `VITE_API_BASE_URL=https://your-render-service.onrender.com`

## Local development

Backend:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```
