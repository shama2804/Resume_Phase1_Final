from web_app import flask_app

print("✅ ROUTES:")
print(flask_app.url_map)

if __name__ == "__main__":
    flask_app.run(debug=True)