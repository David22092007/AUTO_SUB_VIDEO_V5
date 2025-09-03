import os
import random
import sys
import time
import http.client as httplib # For Python 3 compatibility with httplib exceptions

import google_auth_oauthlib.flow
import google.oauth2.credentials
import googleapiclient.discovery
import googleapiclient.errors
import googleapiclient.http
from oauth2client.tools import argparser # Using argparser from oauth2client for consistency
import requests.exceptions # For catching connection errors during authentication

# Explicitly tell the underlying HTTP transport library not to retry, since
# we are handling retry logic ourselves.
# httplib2.RETRIES = 1 # This is for the old httplib2 usage, not directly applicable here but good to note.

# Maximum number of times to retry before giving up.
MAX_RETRIES = 10

# Always retry when these exceptions are raised.
RETRIABLE_EXCEPTIONS = (
    httplib.NotConnected,
    httplib.IncompleteRead,
    httplib.ImproperConnectionState,
    httplib.CannotSendRequest,
    httplib.CannotSendHeader,
    httplib.ResponseNotReady,
    httplib.BadStatusLine,
    IOError # General I/O errors
)

# Always retry when an googleapiclient.errors.HttpError with one of these status
# codes is raised.
RETRIABLE_STATUS_CODES = [500, 502, 503, 504]

# OAuth 2.0 scopes for YouTube Data API
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]          

# File to store/load user credentials
TOKEN_FILE = 'token.json'

# Path to your client secrets file downloaded from Google Cloud Console
CLIENT_SECRETS_FILE = "client_secrets.json" # Changed to common convention

VALID_PRIVACY_STATUSES = ("public", "private", "unlisted")

def authenticate_youtube():
    """Authenticates the user and returns a YouTube API service object."""
    credentials = None

    # Disable OAuthlib's HTTPS verification when running locally.
    # DO NOT leave this setting enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    # Check if a token file exists and load credentials from it
    if os.path.exists(TOKEN_FILE):
        try:
            credentials = google.oauth2.credentials.Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            print("Loaded credentials from token.json")
        except Exception as e:
            print(f"Error loading credentials from token.json: {e}")
            credentials = None # Force re-authentication if token is invalid

    # If no valid credentials, initiate the OAuth flow
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            # Refresh the token if it's expired and a refresh token is available
            print("Credentials expired, attempting to refresh...")
            try:
                credentials.refresh(google.auth.transport.requests.Request())
                print("Credentials refreshed successfully.")
            except Exception as e:
                print(f"Error refreshing credentials: {e}")
                credentials = None # Force full re-authentication if refresh fails
        
        if not credentials or not credentials.valid:
            print("Initiating new authentication flow...")
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            try:
                # This will start a local web server to handle the OAuth redirect
                credentials = flow.run_local_server(port=8080) # Explicitly set port
                print("Authentication successful.")
            except requests.exceptions.ConnectionError as e:
                print(f"\nERROR: Failed to connect to Google's authentication server during token exchange.")
                print(f"This often indicates a network issue, an interfering firewall/antivirus, or outdated SSL certificates.")
                print(f"Please check your internet connection, temporarily disable firewall/antivirus (for testing),")
                print(f"and ensure your Python's SSL certificates are up-to-date (e.g., run 'Install Certificates.command' on Windows).")
                sys.exit(1)
            except Exception as e:
                print(f"An unexpected error occurred during authentication: {e}")
                sys.exit(1)

        # Save the credentials for future use
        with open(TOKEN_FILE, 'w') as token:
            token.write(credentials.to_json())
            print(f"Credentials saved to {TOKEN_FILE}")

    # Build the YouTube API service object
    youtube = googleapiclient.discovery.build(
        "youtube", "v3", credentials=credentials)

    return youtube

def upload_video(youtube, options):
    """Uploads a video to YouTube with retry logic."""
    tags = None
    if options.keywords:
        tags = options.keywords.split(",")

    request_body = {
        "snippet": {
            "categoryId": options.category,
            "title": options.title,
            "description": options.description,
            "tags": tags
        },
        "status": {
            "privacyStatus": options.privacyStatus
        }
    }

    media_file = options.file

    if not os.path.exists(media_file):
        sys.exit(f"Error: Video file '{media_file}' not found.")

    insert_request = youtube.videos().insert(
        part="snippet,status",
        body=request_body,
        media_body=googleapiclient.http.MediaFileUpload(media_file, chunksize=-1, resumable=True)
    )

    response = None
    error = None
    retry = 0

    print(f"Starting upload for '{options.title}' from '{media_file}'...")

    while response is None:
        try:
            status, response = insert_request.next_chunk()
            if status:
                print(f"Upload {int(status.progress() * 100)}%")

            if response is not None:
                if 'id' in response:
                    print(f"Video '{options.title}' uploaded successfully with ID: {response['id']}")
                else:
                    sys.exit(f"The upload failed with an unexpected response: {response}")
        except googleapiclient.errors.HttpError as e:
            if e.resp.status in RETRIABLE_STATUS_CODES:
                error = f"A retriable HTTP error {e.resp.status} occurred:\n{e.content}"
            else:
                raise # Re-raise non-retriable HTTP errors
        except RETRIABLE_EXCEPTIONS as e:
            error = f"A retriable network error occurred: {e}"
        except Exception as e:
            # Catch any other unexpected errors during chunk upload
            print(f"An unexpected error occurred during upload: {e}")
            sys.exit(1)

        if error is not None:
            print(error)
            retry += 1
            if retry > MAX_RETRIES:
                sys.exit(f"Exceeded maximum retries ({MAX_RETRIES}). Giving up on upload.")

            max_sleep = 2 ** retry
            sleep_seconds = random.random() * max_sleep
            print(f"Sleeping {sleep_seconds:.2f} seconds and then retrying...")
            time.sleep(sleep_seconds)
            error = None # Reset error for next retry attempt

def main():
    """Main function to parse arguments, authenticate, and upload video."""
    argparser.add_argument("--file", required=True, help="Path to the video file to upload.")
    argparser.add_argument("--title", default="Uploaded from Python API", help="Title of the video.")
    argparser.add_argument("--description", default="This video was uploaded using the YouTube Data API v3 and Python.", help="Description of the video.")
    argparser.add_argument("--category", default="22",
                           help="Numeric video category ID. See [https://developers.google.com/youtube/v3/docs/videoCategories/list](https://developers.google.com/youtube/v3/docs/videoCategories/list) for a list of categories.")
    argparser.add_argument("--keywords", default="", help="Comma-separated video keywords.")
    argparser.add_argument("--privacyStatus", choices=VALID_PRIVACY_STATUSES,
                           default="private", help="Video privacy status (public, private, or unlisted).")
    args = argparser.parse_args()

    # Ensure client_secrets.json exists
    if not os.path.exists(CLIENT_SECRETS_FILE):
        print(f"ERROR: '{CLIENT_SECRETS_FILE}' not found.")
        print(f"Please download your OAuth 2.0 client secrets JSON file from Google Cloud Console")
        print(f"and save it as '{CLIENT_SECRETS_FILE}' in the same directory as this script.")
        print(f"For more info: [https://developers.google.com/api-client-library/python/guide/aaa_client_secrets](https://developers.google.com/api-client-library/python/guide/aaa_client_secrets)")
        sys.exit(1)

    youtube = authenticate_youtube()
    try:
        upload_video(youtube, args)
    except googleapiclient.errors.HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred during video upload:\n{e.content}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
