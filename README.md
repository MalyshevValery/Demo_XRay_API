# API server for Demo PTD XRay
Backend server for [this repo](https://github.com/MalyshevValery/Demo_XRay_Web)

### Installation
1. `python3 -m venv venv`
2. `pip install -r requirements.txt`
3. Run `python deploy.py` and insert proper link (to get it you can contact me personally)
3. Setup parameters in `settings.py`

### Run
- You can start dev server on `localhost:5000` by running `python wsgi.py`
- You can try test uwsgi configuration by running `uwsgi --ini uwsgi.ini`