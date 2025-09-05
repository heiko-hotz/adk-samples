import os
import uuid
import asyncio
import logging
import contextvars

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

import os, uuid, asyncio, logging, contextvars
import google.cloud.logging as gcl
from google.cloud.logging_v2.handlers import CloudLoggingHandler

# Define context variables. These are essential for managing state in a concurrent
# environment. Each variable's value is local to a specific asynchronous task,
# ensuring that data from different user sessions does not get mixed up.
session_id_var    = contextvars.ContextVar("session_id", default="N/A")
user_id_var       = contextvars.ContextVar("user_id", default="N/A")
invocation_id_var = contextvars.ContextVar("invocation_id", default="N/A")

# This custom logging filter is the core of the solution. It intercepts every log
# message before it's processed by the handler.
class SessionJSONFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Get the existing json_fields dict from the log record or create an empty one.
        jf = getattr(record, "json_fields", {}) or {}
        # Update the dict with the current values from our context variables. The CloudLoggingHandler
        # will use this to populate the `jsonPayload` in Google Cloud Logging.
        jf.update({
            "session_id":    session_id_var.get(),
            "user_id":       user_id_var.get(),
            "invocation_id": invocation_id_var.get(),
        })
        record.json_fields = jf
        # This line modifies the main text of the log message to add the session ID prefix
        # for easy visual scanning in the log viewer.
        record.msg = f"[session-id={session_id_var.get()}] {record.msg}"
        return True

# This function sets up the entire logging pipeline.
def configure_logging():
    client = gcl.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT")) 
    handler = CloudLoggingHandler(client, name="adk-logs")
    # Attach our custom filter to the handler. Every log processed by this handler
    # will now be enriched with the session context.
    handler.addFilter(SessionJSONFilter())

    # Get the root logger specifically for the 'google_adk' library namespace.
    adk_logger = logging.getLogger("google_adk")
    adk_logger.setLevel(logging.DEBUG)
    # Attach our configured handler directly to the ADK logger. This is how we
    # intercept all internal logs from the ADK framework.
    adk_logger.addHandler(handler)
    # Setting propagate to False prevents ADK logs from also being sent to the
    # root logger, which would cause duplicate messages.
    adk_logger.propagate = False

    # Do the same for our application's own logger so it also gets structured logs.
    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)
    app_logger.addHandler(handler)
    app_logger.propagate = False
    
    # Return the handler so we can manually close it later to flush all pending logs.
    return handler

# These functions manage the lifecycle of the context variables for each interaction.
def bind_request_context(session_id, user_id):
    # The .set() method changes the variable's value for the current task and returns a Token object.
    t_s = session_id_var.set(session_id)
    t_u = user_id_var.set(user_id)
    t_i = invocation_id_var.set(str(uuid.uuid4()))
    # These tokens are needed to restore the variables to their previous state after the task is done.
    return t_s, t_u, t_i

def reset_request_context(tokens):
    # Unpack the tokens received from the bind function.
    t_s, t_u, t_i = tokens
    # The .reset() method uses the token to restore the context variable to its exact state
    # before the corresponding .set() call was made. This is essential for cleanup.
    session_id_var.reset(t_s)
    user_id_var.reset(t_u)
    invocation_id_var.reset(t_i)


async def run_interaction(user_id: str, session_id: str, prompt: str, runner: Runner):
    # Bind the context variables at the start of the specific interaction.
    tokens = bind_request_context(session_id=session_id, user_id=user_id)
    app_log = logging.getLogger("app")

    # The try...finally block is a safety mechanism. It guarantees that the context is
    # reset in the `finally` block, even if an error occurs during the agent run.
    try:
        app_log.info("Starting agent run.")
        final_text = "No response"
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        ):
            if event.is_final_response():
                final_text = event.content.parts[0].text.strip()
        app_log.info("Final response received: %s", final_text)
    finally:
        # Clean up the context variables for this interaction, preventing data leaks between tasks.
        reset_request_context(tokens)

async def main():
    handler = configure_logging()

    agent = LlmAgent(
        name="demo_agent",
        model="gemini-2.5-flash",
        instruction="Answer briefly.",
    )
    
    app_name = "adk-logging-mre"
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    user_1_id = "user-001"
    session_1_id = "s-001"
    user_2_id = "user-002"
    session_2_id = "s-002"

    await session_service.create_session(app_name=app_name, user_id=user_1_id, session_id=session_1_id)
    await session_service.create_session(app_name=app_name, user_id=user_2_id, session_id=session_2_id)

    try:
        # `asyncio.create_task` and `asyncio.gather` are used to run the two
        # user interactions concurrently, simulating a multi-user environment.
        task1 = asyncio.create_task(
            run_interaction(user_1_id, session_1_id, "What is the capital of France?", runner)
        )
        await asyncio.sleep(0.1)
        task2 = asyncio.create_task(
            run_interaction(user_2_id, session_2_id, "What is the largest planet?", runner)
        )
        
        await asyncio.gather(task1, task2)
    finally:
        # This is a critical step for short-lived scripts. It tells the handler to send any
        # logs still in its buffer to Google Cloud before the program exits, preventing data loss
        # and shutdown warnings.
        print("Flushing and closing logging handler...")
        handler.close()
        print("Handler closed.")

if __name__ == "__main__":
    asyncio.run(main())