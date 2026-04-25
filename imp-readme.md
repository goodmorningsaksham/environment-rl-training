cell 8.5 logs : Online RL: Checking live environment...
  Connected: https://saksham1771-infinite-dom.hf.space

Collecting online trajectories: 5 episodes x 8 tasks
Max steps per episode: 25
Success threshold: 1+ nodes completed
Protocol: REST (POST /reset + POST /step with action wrapper)
============================================================
    Episode error: HTTPStatusError: Server error '500 Internal Server Error' for url 'https://saksham1771-infinite-d
  Task 1 seed 0: 0/0 nodes
    Episode error: HTTPStatusError: Server error '500 Internal Server Error' for url 'https://saksham1771-infinite-d
  Task 1 seed 1: 0/0 nodes
    Episode error: HTTPStatusError: Server error '500 Internal Server Error' for url 'https://saksham1771-infinite-d
  Task 1 seed 2: 0/0 nodes

hf space logs:
GET /tasks HTTP/1.1" 200 OK
INFO:     10.16.47.45:20473 - "GET /health HTTP/1.1" 200 OK
INFO:     10.16.24.44:49207 - "POST /reset HTTP/1.1" 200 OK
INFO:     10.16.1.233:5045 - "POST /step HTTP/1.1" 500 Internal Server Error
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/protocols/http/httptools_impl.py", line 421, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/middleware/proxy_headers.py", line 56, in __call__
    return await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/applications.py", line 1159, in __call__
    await super().__call__(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/applications.py", line 90, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/errors.py", line 186, in __call__
    raise exc
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/errors.py", line 164, in __call__
    await self.app(scope, receive, _send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/cors.py", line 88, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/exceptions.py", line 63, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/middleware/asyncexitstack.py", line 18, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 660, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 680, in app
    await route.handle(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 134, in app
    await wrap_app_handling_exceptions(app, request)(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 120, in app
    response = await f(request)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 674, in app
    raw_response = await run_endpoint_function(
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 328, in run_endpoint_function
    return await dependant.call(**values)
  File "/usr/local/lib/python3.10/dist-packages/openenv/core/env_server/http_server.py", line 1157, in step
    return await step_handler(request)
  File "/usr/local/lib/python3.10/dist-packages/openenv/core/env_server/http_server.py", line 659, in step_handler
    observation = await _env.step_async(action, **valid_kwargs)
  File "/app/infinite_dom/environment/infinite_dom_env.py", line 90, in step_async
    return await self._step_async(action)
  File "/app/infinite_dom/environment/infinite_dom_env.py", line 137, in _step_async
    assert self._state is not None and self._current_page is not None
AssertionError
INFO:     10.16.24.44:37285 - "POST /reset HTTP/1.1" 200 OK
INFO:     10.16.24.44:4571 - "POST /step HTTP/1.1" 500 Internal Server Error
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/protocols/http/httptools_impl.py", line 421, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/middleware/proxy_headers.py", line 56, in __call__
    return await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/applications.py", line 1159, in __call__
    await super().__call__(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/applications.py", line 90, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/errors.py", line 186, in __call__
    raise exc
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/errors.py", line 164, in __call__
    await self.app(scope, receive, _send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/cors.py", line 88, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/exceptions.py", line 63, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/middleware/asyncexitstack.py", line 18, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 660, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 680, in app
    await route.handle(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 134, in app
    await wrap_app_handling_exceptions(app, request)(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 120, in app
    response = await f(request)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 674, in app
    raw_response = await run_endpoint_function(
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 328, in run_endpoint_function
    return await dependant.call(**values)
  File "/usr/local/lib/python3.10/dist-packages/openenv/core/env_server/http_server.py", line 1157, in step
    return await step_handler(request)
  File "/usr/local/lib/python3.10/dist-packages/openenv/core/env_server/http_server.py", line 659, in step_handler
    observation = await _env.step_async(action, **valid_kwargs)
  File "/app/infinite_dom/environment/infinite_dom_env.py", line 90, in step_async
    return await self._step_async(action)
  File "/app/infinite_dom/environment/infinite_dom_env.py", line 137, in _step_async
    assert self._state is not None and self._current_page is not None
AssertionError
INFO:     10.16.47.45:13612 - "POST /reset HTTP/1.1" 200 OK
INFO:     10.16.27.106:45399 - "POST /step HTTP/1.1" 500 Internal Server Error
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/protocols/http/httptools_impl.py", line 421, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/middleware/proxy_headers.py", line 56, in __call__
    return await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/applications.py", line 1159, in __call__
    await super().__call__(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/applications.py", line 90, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/errors.py", line 186, in __call__
    raise exc
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/errors.py", line 164, in __call__
    await self.app(scope, receive, _send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/cors.py", line 88, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/exceptions.py", line 63, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/middleware/asyncexitstack.py", line 18, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 660, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 680, in app
    await route.handle(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 134, in app
    await wrap_app_handling_exceptions(app, request)(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 120, in app
    response = await f(request)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 674, in app
    raw_response = await run_endpoint_function(
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 328, in run_endpoint_function
    return await dependant.call(**values)
  File "/usr/local/lib/python3.10/dist-packages/openenv/core/env_server/http_server.py", line 1157, in step
    return await step_handler(request)
  File "/usr/local/lib/python3.10/dist-packages/openenv/core/env_server/http_server.py", line 659, in step_handler
    observation = await _env.step_async(action, **valid_kwargs)
  File "/app/infinite_dom/environment/infinite_dom_env.py", line 90, in step_async
    return await self._step_async(action)
  File "/app/infinite_dom/environment/infinite_dom_env.py", line 137, in _step_async
    assert self._state is not None and self._current_page is not None
AssertionError
INFO:     10.16.1.233:6056 - "POST /reset HTTP/1.1" 200 OK
 
