import os, sys
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

def setup_otel(app):
    # ✅ Skip OTel during pytest to avoid exporter writing to closed stdout
    if "pytest" in sys.modules or os.getenv("DISABLE_OTEL") == "1":
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "local-first-ai-api")
    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # 输出到控制台（最简单、立刻能看到）
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # 自动埋点：FastAPI + httpx
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
