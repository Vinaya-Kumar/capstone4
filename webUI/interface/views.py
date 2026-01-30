
from django.shortcuts import render
from interface.ray_tasks.ray_init import init_ray

def index(request):
    context = {}

    if request.method == "POST":
        model = request.POST.get("model")
        tasks = request.POST.getlist("task")
        uploaded_file = request.FILES.get("input_file")

        raw = uploaded_file.read() if uploaded_file else b""
        try:
            text_input = raw.decode("utf-8")
        except Exception:
            text_input = raw.decode("utf-8", errors="replace")

        lines = [ln.strip() for ln in text_input.splitlines() if ln.strip()]

        if not tasks:
            context["error"] = "Please select at least one task."
            context["selected_model"] = model
            return render(request, "interface/index.html", context)

        # Keep your init (ok). Your ensure_ray() also protects you, so either is fine.
        init_ray()

        results = {}

        if "next_token" in tasks:
            from interface.ray_tasks.next_token import run_next_token
            outputs = run_next_token(lines, model)
            # Make it easy for template to render
            results["next_token"] = [{"input": inp, "output": out} for inp, out in zip(lines, outputs)]

        if "embeddings" in tasks:
            from interface.ray_tasks.embeddings import run_embeddings
            embs = run_embeddings(lines, model)
            results["embeddings"] = [
                {"text": t, "dim": len(e), "preview": e[:10]}
                for t, e in zip(lines, embs)
            ]

        context["results"] = results
        context["selected_model"] = model
        context["selected_tasks"] = tasks

    return render(request, "interface/index.html", context)

