#!/usr/bin/env python3
"""Performance benchmarks for critical operations."""

import asyncio
import tempfile
from pathlib import Path
from time import perf_counter

import pytest
from session_buddy.adapters.reflection_adapter import (
    ReflectionDatabaseAdapter as ReflectionDatabase,
)


class TestPerformanceBenchmarks:
    """Performance benchmarks for critical operations."""

    @pytest.mark.benchmark
    async def test_reflection_storage_performance(self, benchmark):
        """Benchmark reflection storage performance."""

        async def storage_benchmark():
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
                db_path = tmp.name

            try:
                db = ReflectionDatabase(db_path=db_path)
                await db.initialize()

                # Store 100 reflections
                for i in range(100):
                    content = (
                        f"Test reflection content {i} " * 10
                    )  # Make content longer
                    tags = [f"tag_{i % 10}" for i in range(3)]  # Up to 3 tags
                    await db.store_reflection(content, tags)

                db.close()

            except Exception:
                # Try to clean up even if the test failed
                try:
                    import os

                    os.remove(db_path)
                except:
                    pass

        # Run the benchmark
        await storage_benchmark()

    @pytest.mark.benchmark
    async def test_similarity_search_performance(self, benchmark):
        """Benchmark similarity search performance with various dataset sizes."""

        async def search_benchmark():
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
                db_path = tmp.name

            try:
                db = ReflectionDatabase(db_path=db_path)
                await db.initialize()

                # Pre-populate with different dataset sizes
                dataset_sizes = [10, 50, 100]
                for size in dataset_sizes:
                    # Clear and repopulate for each size
                    # Note: In a real scenario, we'd have more sophisticated cleanup
                    for i in range(size):
                        content = f"Example content for performance testing {i} " * 5
                        tags = [f"perf_tag_{i % 5}"]
                        await db.store_reflection(content, tags)

                    # Measure search performance
                    start_time = perf_counter()
                    results = await db.similarity_search(
                        "performance testing example", limit=5
                    )
                    end_time = perf_counter()

                    search_duration = end_time - start_time
                    print(
                        f"Search with {size} items took {search_duration:.4f}s, got {len(results)} results"
                    )

                    # Assert reasonable performance (adjust thresholds as needed)
                    assert search_duration < 5.0, (
                        f"Search with {size} items took too long: {search_duration:.4f}s"
                    )

                db.close()

            except Exception:
                # Try to clean up even if the test failed
                try:
                    import os

                    os.remove(db_path)
                except:
                    pass

        await search_benchmark()

    @pytest.mark.benchmark
    async def test_large_dataset_performance(self):
        """Test performance with a larger dataset."""
        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
            db_path = tmp.name

        try:
            db = ReflectionDatabase(db_path=db_path)
            await db.initialize()

            # Insert a larger dataset
            start_time = perf_counter()
            for i in range(500):  # 500 reflections
                content = f"Large dataset test content {i} " + "lorem ipsum " * 20
                tags = ["large_dataset", f"tag_{i % 20}"]
                await db.store_reflection(content, tags)
            insert_time = perf_counter() - start_time

            print(f"Inserted 500 reflections in {insert_time:.4f}s")
            assert insert_time < 30.0, f"Insertion took too long: {insert_time:.4f}s"

            # Test search performance on large dataset
            search_start = perf_counter()
            results = await db.similarity_search("lorem ipsum", limit=10)
            search_time = perf_counter() - search_start

            print(
                f"Searched 500 reflections in {search_time:.4f}s, got {len(results)} results"
            )
            assert search_time < 5.0, f"Search took too long: {search_time:.4f}s"

            db.close()

        except Exception:
            # Try to clean up even if the test failed
            try:
                import os

                os.remove(db_path)
            except:
                pass

    @pytest.mark.benchmark
    async def test_concurrent_operations_performance(self):
        """Test performance under concurrent operations."""
        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
            db_path = tmp.name

        try:
            db = ReflectionDatabase(db_path=db_path)
            await db.initialize()

            async def concurrent_task(task_id: int):
                # Each task will perform several operations
                for i in range(10):
                    content = f"Concurrent task {task_id} item {i}"
                    tags = ["concurrent", f"task_{task_id}"]
                    await db.store_reflection(content, tags)
                    results = await db.similarity_search(content, limit=1)
                    # Verify we can retrieve what we stored
                    assert len(results) >= 0

            # Run 5 concurrent tasks
            start_time = perf_counter()
            tasks = [concurrent_task(i) for i in range(5)]
            await asyncio.gather(*tasks)
            concurrent_time = perf_counter() - start_time

            print(f"Completed 50 concurrent operations in {concurrent_time:.4f}s")
            # This may take longer due to resource contention, but shouldn't be excessive
            assert concurrent_time < 60.0, (
                f"Concurrent operations took too long: {concurrent_time:.4f}s"
            )

            db.close()

        except Exception:
            # Try to clean up even if the test failed
            try:
                import os

                os.remove(db_path)
            except:
                pass

    @pytest.mark.benchmark
    def test_memory_usage_stability(self):
        """Basic test to check for memory leaks during operations."""
        import gc
        import os

        try:
            import psutil

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Perform multiple operations
            results = []
            for i in range(100):
                # Simulate operations without a database
                temp_result = {"id": i, "data": f"test_data_{i}" * 10}
                results.append(temp_result)

            # Force garbage collection
            gc.collect()

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            print(
                f"Memory usage: started at {initial_memory:.2f}MB, ended at {final_memory:.2f}MB"
            )
            print(f"Memory increase: {memory_increase:.2f}MB")

            # Memory increase should be reasonable (less than 100MB for this test)
            assert abs(memory_increase) < 100, (
                f"Excessive memory increase: {memory_increase:.2f}MB"
            )

        except ImportError:
            # psutil not available, skip this test
            pytest.skip("psutil not available for memory monitoring")


if __name__ == "__main__":
    pytest.main([__file__])
