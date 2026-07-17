"""Fixed prompt sets.

PARITY: small, diverse; used for correctness gates (greedy outputs must match
across single-process reference, EP, and EP+balancer).

SKEW: larger, single-domain (code-flavored); used for the load-balancing
benchmark. A homogeneous domain concentrates router traffic on a subset of
experts, which is exactly the hot-expert regime the balancer targets.
"""

PARITY = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "In 1969, the Apollo 11 mission",
    "SELECT name, age FROM users WHERE",
    "The mitochondria is the powerhouse of",
    "Once upon a time, in a quiet village,",
    "To reverse a linked list in C, you",
    "El clima de hoy en Madrid es",
]

_CODE_SNIPPETS = [
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)\n",
    "class LinkedList:\n    def __init__(self):\n        self.head = None\n\n    def append(self, value):\n        node = Node(value)\n        if self.head is None:\n            self.head = node\n            return\n        cur = self.head\n        while cur.next:\n            cur = cur.next\n        cur.next = node\n",
    "import json\n\nwith open('config.json') as f:\n    config = json.load(f)\n\nfor key, value in config.items():\n    print(f'{key}: {value}')\n",
    "SELECT u.name, COUNT(o.id) AS order_count\nFROM users u\nJOIN orders o ON o.user_id = u.id\nWHERE o.created_at > '2024-01-01'\nGROUP BY u.name\nHAVING COUNT(o.id) > 5\nORDER BY order_count DESC;\n",
    "async function fetchData(url) {\n    const response = await fetch(url);\n    if (!response.ok) {\n        throw new Error(`HTTP ${response.status}`);\n    }\n    return response.json();\n}\n",
    "#include <stdio.h>\n\nint main(void) {\n    int arr[10];\n    for (int i = 0; i < 10; i++) {\n        arr[i] = i * i;\n    }\n    for (int i = 0; i < 10; i++) {\n        printf(\"%d\\n\", arr[i]);\n    }\n    return 0;\n}\n",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    "CREATE TABLE orders (\n    id SERIAL PRIMARY KEY,\n    user_id INTEGER REFERENCES users(id),\n    total NUMERIC(10, 2) NOT NULL,\n    created_at TIMESTAMP DEFAULT NOW()\n);\n",
    "import torch\nimport torch.nn as nn\n\nclass MLP(nn.Module):\n    def __init__(self, dim, hidden):\n        super().__init__()\n        self.fc1 = nn.Linear(dim, hidden)\n        self.fc2 = nn.Linear(hidden, dim)\n\n    def forward(self, x):\n        return self.fc2(torch.relu(self.fc1(x)))\n",
    "{\n  \"name\": \"mini-ep\",\n  \"version\": \"0.1.0\",\n  \"dependencies\": {\n    \"torch\": \"^2.0.0\",\n    \"fastapi\": \"^0.100.0\"\n  },\n  \"scripts\": {\n    \"serve\": \"uvicorn app:main --port 8000\"\n  }\n}\n",
    "for file in *.log; do\n    gzip -9 \"$file\"\n    mv \"$file.gz\" archive/\ndone\n\nfind archive/ -mtime +30 -delete\n",
    "def dijkstra(graph, source):\n    dist = {v: float('inf') for v in graph}\n    dist[source] = 0\n    pq = [(0, source)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]:\n            continue\n        for v, w in graph[u]:\n            if dist[u] + w < dist[v]:\n                dist[v] = dist[u] + w\n                heapq.heappush(pq, (dist[v], v))\n    return dist\n",
]

# Repeat the snippet pool to form the benchmark workload: a sustained stream of
# same-domain traffic (what a code-assistant deployment looks like).
SKEW = _CODE_SNIPPETS * 4
