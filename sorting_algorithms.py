#!/usr/bin/env python3
"""
排序算法演示集合
Sorting Algorithms Demonstration Suite

包含常见排序算法的实现、性能测试和可视化
"""

import copy
import random
import time
from typing import Any

import matplotlib.pyplot as plt


class SortingAlgorithms:
    """排序算法集合类"""

    def __init__(self):
        self.comparisons = 0
        self.swaps = 0

    def reset_counters(self):
        """重置计数器"""
        self.comparisons = 0
        self.swaps = 0

    # 1. 冒泡排序 (Bubble Sort)
    def bubble_sort(self, arr: list[Any]) -> list[Any]:
        """冒泡排序 - O(n²)"""
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                self.comparisons += 1
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    self.swaps += 1
        return arr

    # 2. 选择排序 (Selection Sort)
    def selection_sort(self, arr: list[Any]) -> list[Any]:
        """选择排序 - O(n²)"""
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                self.comparisons += 1
                if arr[j] < arr[min_idx]:
                    min_idx = j
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
                self.swaps += 1
        return arr

    # 3. 插入排序 (Insertion Sort)
    def insertion_sort(self, arr: list[Any]) -> list[Any]:
        """插入排序 - O(n²)"""
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                self.comparisons += 1
                arr[j + 1] = arr[j]
                self.swaps += 1
                j -= 1
            arr[j + 1] = key
        return arr

    # 4. 快速排序 (Quick Sort)
    def quick_sort(self, arr: list[Any]) -> list[Any]:
        """快速排序 - O(n log n) 平均"""
        def partition(low: int, high: int) -> int:
            pivot = arr[high]
            i = low - 1

            for j in range(low, high):
                self.comparisons += 1
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
                    self.swaps += 1

            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            self.swaps += 1
            return i + 1

        def quick_sort_recursive(low: int, high: int):
            if low < high:
                pi = partition(low, high)
                quick_sort_recursive(low, pi - 1)
                quick_sort_recursive(pi + 1, high)

        quick_sort_recursive(0, len(arr) - 1)
        return arr

    # 5. 归并排序 (Merge Sort)
    def merge_sort(self, arr: list[Any]) -> list[Any]:
        """归并排序 - O(n log n)"""
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = self.merge_sort(arr[:mid])
        right = self.merge_sort(arr[mid:])

        return self.merge(left, right)

    def merge(self, left: list[Any], right: list[Any]) -> list[Any]:
        """归并排序的合并操作"""
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            self.comparisons += 1
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result

    # 6. 堆排序 (Heap Sort)
    def heap_sort(self, arr: list[Any]) -> list[Any]:
        """堆排序 - O(n log n)"""
        def heapify(n: int, i: int):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2

            if left < n:
                self.comparisons += 1
                if arr[left] > arr[largest]:
                    largest = left

            if right < n:
                self.comparisons += 1
                if arr[right] > arr[largest]:
                    largest = right

            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.swaps += 1
                heapify(n, largest)

        n = len(arr)

        # 构建最大堆
        for i in range(n // 2 - 1, -1, -1):
            heapify(n, i)

        # 逐个提取元素
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            self.swaps += 1
            heapify(i, 0)

        return arr

    # 7. 希尔排序 (Shell Sort)
    def shell_sort(self, arr: list[Any]) -> list[Any]:
        """希尔排序 - O(n^1.3) 平均"""
        n = len(arr)
        gap = n // 2

        while gap > 0:
            for i in range(gap, n):
                temp = arr[i]
                j = i
                while j >= gap and arr[j - gap] > temp:
                    self.comparisons += 1
                    arr[j] = arr[j - gap]
                    self.swaps += 1
                    j -= gap
                arr[j] = temp
            gap //= 2

        return arr

    # 8. 计数排序 (Counting Sort)
    def counting_sort(self, arr: list[int]) -> list[int]:
        """计数排序 - O(n + k) 适用于整数"""
        if not arr:
            return arr

        max_val = max(arr)
        min_val = min(arr)
        range_val = max_val - min_val + 1

        count = [0] * range_val
        output = [0] * len(arr)

        # 统计元素出现次数
        for num in arr:
            count[num - min_val] += 1

        # 计算累积计数
        for i in range(1, len(count)):
            count[i] += count[i - 1]

        # 构建输出数组
        for num in reversed(arr):
            output[count[num - min_val] - 1] = num
            count[num - min_val] -= 1

        return output

    # 9. 基数排序 (Radix Sort)
    def radix_sort(self, arr: list[int]) -> list[int]:
        """基数排序 - O(d * (n + k))"""
        if not arr:
            return arr

        def counting_sort_for_radix(arr: list[int], exp: int) -> list[int]:
            n = len(arr)
            output = [0] * n
            count = [0] * 10

            # 统计数字出现次数
            for num in arr:
                index = (num // exp) % 10
                count[index] += 1

            # 计算累积计数
            for i in range(1, 10):
                count[i] += count[i - 1]

            # 构建输出数组
            for num in reversed(arr):
                index = (num // exp) % 10
                output[count[index] - 1] = num
                count[index] -= 1

            return output

        max_val = max(arr)
        exp = 1

        while max_val // exp > 0:
            arr = counting_sort_for_radix(arr, exp)
            exp *= 10

        return arr

    # 10. 桶排序 (Bucket Sort)
    def bucket_sort(self, arr: list[float], bucket_size: int = 5) -> list[float]:
        """桶排序 - O(n + k) 平均"""
        if not arr:
            return arr

        min_val = min(arr)
        max_val = max(arr)

        # 计算桶的数量
        bucket_count = (max_val - min_val) // bucket_size + 1
        buckets = [[] for _ in range(bucket_count)]

        # 将元素分配到桶中
        for num in arr:
            bucket_index = int((num - min_val) // bucket_size)
            buckets[bucket_index].append(num)

        # 对每个桶进行排序并合并
        sorted_arr = []
        for bucket in buckets:
            if bucket:
                sorted_bucket = sorted(bucket)
                sorted_arr.extend(sorted_bucket)

        return sorted_arr


class SortingBenchmark:
    """排序算法性能测试类"""

    def __init__(self):
        self.sorter = SortingAlgorithms()
        self.algorithms = {
            '冒泡排序': self.sorter.bubble_sort,
            '选择排序': self.sorter.selection_sort,
            '插入排序': self.sorter.insertion_sort,
            '快速排序': self.sorter.quick_sort,
            '归并排序': self.sorter.merge_sort,
            '堆排序': self.sorter.heap_sort,
            '希尔排序': self.sorter.shell_sort,
            '计数排序': self.sorter.counting_sort,
            '基数排序': self.sorter.radix_sort,
            '桶排序': self.sorter.bucket_sort
        }

    def generate_test_data(self, size: int, data_type: str = 'random') -> list[Any]:
        """生成测试数据"""
        if data_type == 'random':
            return [random.randint(0, 1000) for _ in range(size)]
        elif data_type == 'sorted':
            return list(range(size))
        elif data_type == 'reversed':
            return list(range(size, 0, -1))
        elif data_type == 'nearly_sorted':
            arr = list(range(size))
            # 随机交换几对元素
            for _ in range(size // 10):
                i, j = random.sample(range(size), 2)
                arr[i], arr[j] = arr[j], arr[i]
            return arr
        else:
            return [random.randint(0, 1000) for _ in range(size)]

    def benchmark_algorithm(self, algorithm_name: str, data: list[Any]) -> dict:
        """测试单个算法性能"""
        self.sorter.reset_counters()
        test_data = copy.deepcopy(data)

        start_time = time.time()

        try:
            if algorithm_name == '计数排序' or algorithm_name == '基数排序':
                # 确保数据是整数
                test_data = [int(x) for x in test_data]
            elif algorithm_name == '桶排序':
                # 桶排序要求数据是浮点数或在合理范围内
                test_data = [float(x) for x in test_data]

            self.algorithms[algorithm_name](test_data)
            end_time = time.time()

            return {
                'algorithm': algorithm_name,
                'time': end_time - start_time,
                'comparisons': self.sorter.comparisons,
                'swaps': self.sorter.swaps,
                'success': True
            }
        except Exception as e:
            return {
                'algorithm': algorithm_name,
                'time': float('inf'),
                'comparisons': 0,
                'swaps': 0,
                'success': False,
                'error': str(e)
            }

    def run_benchmark(self, sizes: list[int], data_types: list[str] = None) -> dict:
        """运行完整基准测试"""
        if data_types is None:
            data_types = ['random', 'sorted', 'reversed', 'nearly_sorted']

        results = {}

        for size in sizes:
            results[size] = {}
            print(f"\n测试数据大小: {size}")
            print("-" * 50)

            for data_type in data_types:
                print(f"数据类型: {data_type}")
                test_data = self.generate_test_data(size, data_type)
                results[size][data_type] = {}

                for algorithm_name in self.algorithms.keys():
                    result = self.benchmark_algorithm(algorithm_name, test_data)
                    results[size][data_type][algorithm_name] = result

                    if result['success']:
                        print(f"  {algorithm_name}: {result['time']:.6f}s, "
                              f"比较: {result['comparisons']}, 交换: {result['swaps']}")
                    else:
                        print(f"  {algorithm_name}: 失败 - {result.get('error', '未知错误')}")

        return results

    def visualize_results(self, results: dict, save_plot: bool = True):
        """可视化测试结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('排序算法性能比较', fontsize=16)

        data_types = list(results[list(results.keys())[0]].keys())
        sizes = list(results.keys())

        for idx, data_type in enumerate(data_types):
            ax = axes[idx // 2, idx % 2]

            for algorithm_name in self.algorithms.keys():
                times = []
                for size in sizes:
                    result = results[size][data_type][algorithm_name]
                    if result['success'] and result['time'] != float('inf'):
                        times.append(result['time'])
                    else:
                        times.append(None)

                # 过滤掉None值
                valid_times = [(size, time) for size, time in zip(sizes, times) if time is not None]
                if valid_times:
                    valid_sizes, valid_times = zip(*valid_times)
                    ax.plot(valid_sizes, valid_times, marker='o', label=algorithm_name)

            ax.set_xlabel('数据大小')
            ax.set_ylabel('时间 (秒)')
            ax.set_title(f'{data_type} 数据')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        plt.tight_layout()

        if save_plot:
            plt.savefig('sorting_algorithms_performance.png', dpi=300, bbox_inches='tight')
            print("性能图表已保存为 'sorting_algorithms_performance.png'")

        plt.show()


def main():
    """主函数 - 演示所有排序算法"""
    print("=" * 60)
    print("排序算法演示集合")
    print("Sorting Algorithms Demonstration Suite")
    print("=" * 60)

    # 创建基准测试实例
    benchmark = SortingBenchmark()

    # 测试数据大小
    test_sizes = [100, 500, 1000, 2000]

    print("\n开始性能测试...")
    results = benchmark.run_benchmark(test_sizes)

    print("\n生成性能图表...")
    benchmark.visualize_results(results)

    # 简单演示
    print("\n" + "=" * 60)
    print("简单演示 - 对小数组进行排序")
    print("=" * 60)

    demo_data = [64, 34, 25, 12, 22, 11, 90]
    print(f"原始数据: {demo_data}")

    sorter = SortingAlgorithms()
    for algorithm_name, algorithm_func in benchmark.algorithms.items():
        test_data = copy.deepcopy(demo_data)

        try:
            if algorithm_name in ['计数排序', '基数排序']:
                test_data = [int(x) for x in test_data]
            elif algorithm_name == '桶排序':
                test_data = [float(x) for x in test_data]

            sorter.reset_counters()
            result = algorithm_func(test_data)
            print(f"{algorithm_name}: {result}")
        except Exception as e:
            print(f"{algorithm_name}: 错误 - {e}")

    print("\n演示完成！")


if __name__ == "__main__":
    main()
