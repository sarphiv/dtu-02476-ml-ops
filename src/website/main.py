import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()


@app.get("/")
async def main():
    return HTMLResponse(
        status_code=200,
        content=f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="upgrade-insecure-requests">
    <title>What is it?</title>

    <style>
        body {{
            font-family: Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;

            background-color: #111;
        }}

        #file-area {{
            color: white;
            border: 2px dashed #80f;
            border-radius: 20px;
            width: 300px;
            padding: 3em;
            text-align: center;
        }}

        #file-form {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}

        #file-elem {{
            display: none;
        }}

        #submit-btn {{
            display: inline-block;
            font-size: 2em;
            padding: 10px 20px;
            background-color: #80f;
            color: white;
            border-radius: 5px;
            cursor: pointer;
        }}

        #loader {{
            display: none;

            border: 32px solid white;
            border-top: 32px solid #80f;
            border-radius: 50%;
            width: 120px;
            height: 120px;
            animation: spin 2s linear infinite;
        }}

        #status {{
            font-size: 3em;
            margin-top: 0.4em;
            margin-bottom: 0.8em;
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

    </style>
</head>
<body>
    <div id="file-area">
        <form id="file-form">
            <div id="status">What is it?</div>
            <input id="file-elem" type="file" accept="image/*" onchange="handleFiles(this.files)">
            <label id="submit-btn" for="file-elem">
                Upload image
            </label>
        </form>
    </div>
    <div id="loader"></div>

    <script>
        function handleFiles(files) {{
            const formData = new FormData();
            formData.append('data', files[0]);

            document.getElementById('file-area').style.display = 'none';
            document.getElementById('loader').style.display = 'block';

            fetch('{os.getenv("INFERENCE_API_URL") or "localhost:8080"}', {{
                method: 'POST',
                body: formData,
                headers: {{
                    "accept": "application/json",
                    "Access-Control-Allow-Origin": "*"
                }}
            }})
            .then(response => response.json())
            .then(data => {{
                console.log(data);

                let sentences = [
                    `The ${{data['predicted-class']}} has arrived!`,
                    `Look! The ${{data['predicted-class']}} is flying!`,
                    `Purchase the new ${{data['predicted-class']}}-inator 9000!`,
                    `I forgot my ${{data['predicted-class']}} at home.`,
                    `Help, my ${{data['predicted-class']}} is drunk.`,
                    `I'm going to the ${{data['predicted-class']}} store, do you want anything?`,
                    `What is that? Oh, it's just my ${{data['predicted-class']}}.`
                ];

                let sentence = sentences[Math.floor(Math.random() * sentences.length)];

                document.getElementById('status').innerHTML = sentence;
            }})
            .catch(error => {{
                console.error(error);
                document.getElementById('status').innerHTML = `Oh noes, something went wrong!`;
            }})
            .finally(() => {{
                document.getElementById('file-area').style.display = 'block';
                document.getElementById('loader').style.display = 'none';
            }});
        }}

        document.getElementById('file-area').ondragover = (event) => {{
            event.preventDefault();
        }};

        document.getElementById('file-area').ondrop = (event) => {{
            event.preventDefault();
            handleFiles(event.dataTransfer.files);
        }};
    </script>
</body>
</html>
        """
    )
