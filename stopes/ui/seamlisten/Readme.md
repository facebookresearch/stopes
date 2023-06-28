![stopes](/stopes/ui/seamlisten/react_app/public/logo.png?raw=true "Seamlisten.")

# Seamlisten

This app is made to explore mining results (.gz archives), whether they are from speech and text, or speech to speech.

## Running the app

For most usecases, only the backend is required

### Backend installation

Install stopes by running `pip install -e '.[speech]'` at the root of the repository.
Then:

```
cd backend
pip install -r requirements.txt
conda install 'ffmpeg<5'
```

### Running the backend

Fastapi uses uvicorn and starlette. To run the app, you can use:

`bash run_backend.sh` or `python main.py`from /backend

If `static/` folder does not already exist in `backend/app/`, create an empty folder before run the bankend app.

### Running the frontend

To benefit from hot reloading, the easiest is to use npm ( `npm start` from /react_app). All source code for the react front end can be found in react_app/. It should be ready to talk to the running backend correctly, without needing to build.

### Build the frontend

The frontend can be built with `npm run build`. Copying the built app from `react_app/build/static/` to `/backend/app/static/` to be automatically discovered and served by fastapi.
