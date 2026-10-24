from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from flask_cors import CORS

from web.routes_main import bp_main
from web.routes_dataset import bp_dataset
from web.routes_pipeline import bp_pipeline
from web.routes_training import bp_training
from web.routes_predict import bp_predict
from web.routes_results import bp_results
from web.routes_experiments import bp_experiments
from web.helpers import MAX_UPLOAD_BYTES


def create_app() -> Flask:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, "templates"),
        static_folder=os.path.join(base_dir, "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["SECRET_KEY"] = os.environ.get(
        "SENSOR_WEB_SECRET",
        "dev-secret-change-me-for-production",
    )

    CORS(app)

    app.register_blueprint(bp_main)
    app.register_blueprint(bp_dataset, url_prefix="/dataset")
    app.register_blueprint(bp_pipeline, url_prefix="/pipeline")
    app.register_blueprint(bp_training, url_prefix="/training")
    app.register_blueprint(bp_predict,  url_prefix="/predict")
    app.register_blueprint(bp_results,  url_prefix="/results")
    app.register_blueprint(bp_experiments, url_prefix="/experiments")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
