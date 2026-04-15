/**
 * FT Table Component v1.6.20260331-5
 * 版本号说明：主版本。次版本。日期（YYYYMMDD）-修订号
 *
 * 新功能 v1.6：
 * - 通配符 '*' 支持：显示所有未指定的字段
 *   用法：:cols="['代码', '涨跌幅', '*']"
 *   位置决定其他字段的显示位置
 *
 * - 通配符排除支持：{ field: '*', exclude: ['字段1', '字段2'] }
 *   用法：:cols="[
 *     { field: '名称', title: '股票信息', slot: 'cell-组合' },
 *     { field: '*', exclude: ['代码', '名称'] }
 *   ]"
 *   排除的字段不显示，但数据仍可在插槽中通过 row 访问
 *
 * 修复 v1.6.20260331-3：
 * - 热力图范围计算改为基于完整数据，不受分页、排序影响
 * - excludeRows 索引改为全局索引（之前是分页索引）
 *
 * 优化 v1.6.20260331-4：
 * - 移除多余的 totalRecords computed，直接使用 props.data.length
 *
 * 优化 v1.6.20260331-5：
 * - 移除字段过滤逻辑，支持操作列等虚拟列（字段不存在于数据但有插槽的列）
 * */

const FtTable = {
  name: 'FtTable',

  props: {
    data: {
      type: Array,
      default: () => []
    },
    cols: {
      type: Array,
      default: () => []
    },
    page: {
      type: [Object, Boolean],
      default: () => ({ size: 10, options: [10, 20, 50, 100] })
    },
    pagination: {
      type: [Object, Boolean],
      default: false
    },
    freeze: {
      type: Object,
      default: () => ({ left: 0, right: 0 })
    },
    heatmap: {
      type: [Object, Boolean],
      default: false
    },
    emptyText: {
      type: String,
      default: '暂无数据'
    },
    resetPage: {
      type: Boolean,
      default: true
    },
    scrollButton: {
      type: Boolean,
      default: true
    }
  },

  setup(props, { slots }) {
    const { ref, computed, onMounted, onUnmounted, watch, nextTick } = Vue;

    // ========== 响应式数据 ==========
    const pageConfig = computed(() => {
      if (props.page === false) return false;
      return props.page || props.pagination;
    });

    const currentPage = ref(1);
    const pageSize = ref(props.page?.size || props.pagination?.size || props.page?.pageSize || props.pagination?.pageSize || 10);
    const multiSort = ref([]);
    const tableContainer = ref(null);
    const resizeObserver = ref(null);

    // ========== 计算属性 ==========

    // 排序后的数据
    const sortedData = computed(() => {
      if (multiSort.value.length === 0) {
        return props.data;
      }

      return [...props.data].sort((a, b) => {
        for (const sort of multiSort.value) {
          const v1 = a[sort.field];
          const v2 = b[sort.field];
          if (v1 !== v2) {
            return sort.order === 'asc' ? (v1 > v2 ? 1 : -1) : (v1 < v2 ? 1 : -1);
          }
        }
        return 0;
      });
    });

    // 分页后的数据
    const paginatedData = computed(() => {
      if (!pageConfig.value) return sortedData.value;
      const start = (currentPage.value - 1) * pageSize.value;
      return sortedData.value.slice(start, start + pageSize.value);
    });

    // 总页数
    const totalPages = computed(() => {
      if (!pageConfig.value) return 1;
      return Math.ceil(props.data.length / pageSize.value) || 1;
    });

    // 显示的列（处理 cols 格式）
    // cols 可选：不传或无效时自动推断所有字段
    // 支持通配符 '*'：表示所有未指定的字段，位置决定其他字段的显示位置
    // 支持对象格式 { field: '*', exclude: ['字段1', '字段2'] }：排除指定字段
    const displayCols = computed(() => {
      // 检查 cols 是否有效数组
      const isValidCols = Array.isArray(props.cols) && props.cols.length > 0;

      // 从数据获取所有字段
      const allFields = (props.data && props.data.length > 0)
        ? Object.keys(props.data[0])
        : [];

      if (!isValidCols) {
        // cols 无效或为空：从数据自动推断所有字段
        return allFields.map(key => ({
          field: key,
          title: key,
          slot: 'cell-' + key
        }));
      }

      // 查找通配符 '*' 的位置（字符串 '*' 或对象 { field: '*' }）
      const wildcardIndex = props.cols.findIndex(col => {
        if (col === '*') return true;
        if (typeof col === 'object' && col !== null && col.field === '*') return true;
        return false;
      });

      if (wildcardIndex === -1) {
        // 无通配符：只显示 cols 中的字段
        return props.cols.map(col => {
          if (typeof col === 'string') {
            return { field: col, title: col, slot: 'cell-' + col };
          }
          const slotName = col.slot || (col.field ? 'cell-' + col.field : null);
          return { ...col, slot: slotName };
        });
      }

      // 有通配符：收集已指定的字段
      const specifiedFields = new Set();
      const specifiedCols = [];

      props.cols.forEach((col, index) => {
        if (col === '*') return;
        if (typeof col === 'object' && col !== null && col.field === '*') return;
        const field = typeof col === 'string' ? col : col.field;
        if (field) {
          specifiedFields.add(field);
          if (typeof col === 'string') {
            specifiedCols.push({ field: col, title: col, slot: 'cell-' + col, _index: index });
          } else {
            const slotName = col.slot || (col.field ? 'cell-' + col.field : null);
            specifiedCols.push({ ...col, title: col.title || field, slot: slotName, _index: index });
          }
        }
      });

      // 获取通配符配置中的 exclude 列表
      const wildcardCol = props.cols[wildcardIndex];
      const excludeFields = new Set();
      if (typeof wildcardCol === 'object' && wildcardCol.exclude) {
        wildcardCol.exclude.forEach(f => excludeFields.add(f));
      }

      // 获取未指定的字段（保持数据中的顺序），并排除 exclude 中的字段
      const unspecifiedFields = allFields.filter(f => !specifiedFields.has(f) && !excludeFields.has(f));

      // 构建最终列配置：按 cols 顺序，'*' 替换为未指定字段
      const result = [];
      props.cols.forEach((col, index) => {
        if (col === '*' || (typeof col === 'object' && col !== null && col.field === '*')) {
          // 插入所有未指定的字段
          unspecifiedFields.forEach(f => {
            result.push({ field: f, title: f, slot: 'cell-' + f });
          });
        } else {
          // 查找对应的列配置
          const colConfig = specifiedCols.find(c => c._index === index);
          if (colConfig) {
            const { _index, ...rest } = colConfig;
            result.push(rest);
          }
        }
      });

      return result;
    });

    // 检查是否有自定义插槽
    const hasSlot = (slotName) => {
      return slotName && slots[slotName];
    };

    // 是否有冻结列
    const hasFreeze = computed(() => {
      return props.freeze && (props.freeze.left > 0 || props.freeze.right > 0);
    });

    // 是否有横向滚动条
    const hasHorizontalScroll = computed(() => {
      if (!tableContainer.value) return false;
      return tableContainer.value.scrollWidth > tableContainer.value.clientWidth;
    });

    // 是否启用热力图（全局或列级别）
    const hasHeatmap = computed(() => {
      if (props.heatmap && typeof props.heatmap === 'object') {
        return true;
      }
      return displayCols.value.some(col => col.heatmap);
    });

    // 解析列的热力图配置
    const parseColHeatmap = (col) => {
      if (!col.heatmap) return null;

      if (typeof col.heatmap === 'object') {
        return {
          enabled: true,
          group: col.heatmap.group || col.field,
          colors: col.heatmap.colors || null
        };
      }

      return null;
    };

    // 热力图配置（合并默认值）
    const heatmapConfig = computed(() => {
      if (!props.heatmap || props.heatmap === true) {
        return null;
      }

      const defaults = {
        start: 1,
        end: -1,
        exclude: [],
        excludeRows: [],
        columns: null,
        colors: ['#2196f3', '#fff', '#f44336'],
        axis: 'column'
      };

      return { ...defaults, ...props.heatmap };
    });

    // 热力图范围缓存（支持分组）
    // 基于完整数据计算，不受分页、排序影响
    const heatmapRanges = computed(() => {
      const data = props.data;
      const cols = displayCols.value;

      if (data.length === 0) return {};

      const ranges = {};
      const groupValues = {};

      const hasGlobalHeatmap = props.heatmap && typeof props.heatmap === 'object';
      const hasColHeatmap = cols.some(col => col.heatmap);

      if (!hasGlobalHeatmap && !hasColHeatmap) {
        return ranges;
      }

      const excludeRowsSet = new Set(
        (heatmapConfig.value?.excludeRows || []).map(idx => idx < 0 ? data.length + idx : idx)
      );
      const validRows = data.filter((_, idx) => !excludeRowsSet.has(idx));

      if (hasColHeatmap) {
        cols.forEach(col => {
          const colHeatmap = parseColHeatmap(col);
          if (!colHeatmap) return;

          const groupName = colHeatmap.group;

          if (!groupValues[groupName]) {
            groupValues[groupName] = [];
          }

          validRows.forEach(row => {
            const v = row[col.field];
            if (typeof v === 'number' && !isNaN(v)) {
              groupValues[groupName].push(v);
            }
          });
        });

        Object.keys(groupValues).forEach(groupName => {
          const values = groupValues[groupName];
          if (values.length > 0) {
            ranges[groupName] = {
              min: Math.min(...values),
              max: Math.max(...values)
            };
          }
        });
      }

      if (hasGlobalHeatmap) {
        const config = heatmapConfig.value;

        if (config.columns && config.columns.length > 0) {
          config.columns.forEach(field => {
            const values = validRows
              .map(row => row[field])
              .filter(v => typeof v === 'number' && !isNaN(v));

            if (values.length > 0) {
              ranges[field] = {
                min: Math.min(...values),
                max: Math.max(...values)
              };
            }
          });
        } else {
          const config = heatmapConfig.value;
          const totalCols = cols.length;
          const start = config.start > 0 ? config.start - 1 : totalCols + config.start;
          const end = config.end > 0 ? config.end - 1 : totalCols + config.end;
          const excludeSet = new Set((config.exclude || []).map(v => {
            return v > 0 ? v - 1 : totalCols + v;
          }));

          const heatmapCols = [];
          for (let i = start; i <= end; i++) {
            if (i >= 0 && i < totalCols && !excludeSet.has(i)) {
              heatmapCols.push(cols[i]);
            }
          }

          if (config.axis === 'column') {
            heatmapCols.forEach(col => {
              if (col.heatmap) return;

              const values = validRows
                .map(row => row[col.field])
                .filter(v => typeof v === 'number' && !isNaN(v));

              if (values.length > 0) {
                ranges[col.field] = {
                  min: Math.min(...values),
                  max: Math.max(...values)
                };
              }
            });
          } else if (config.axis === 'table') {
            const allValues = [];
            heatmapCols.forEach(col => {
              if (col.heatmap) return;
              validRows.forEach(row => {
                const v = row[col.field];
                if (typeof v === 'number' && !isNaN(v)) {
                  allValues.push(v);
                }
              });
            });

            if (allValues.length > 0) {
              const globalRange = {
                min: Math.min(...allValues),
                max: Math.max(...allValues)
              };
              heatmapCols.forEach(col => {
                if (!col.heatmap) {
                  ranges[col.field] = globalRange;
                }
              });
            }
          }
        }
      }

      return ranges;
    });

    // 分页选项
    const pageSizeOptions = computed(() => {
      return props.page?.options || props.pagination?.options || props.page?.pageSizeOptions || props.pagination?.pageSizeOptions || [10, 20, 50, 100];
    });

    // 生成显示的页码数组（首页 上一页 1 2 3 4 5 末页）
    const visiblePages = computed(() => {
      const total = totalPages.value;
      const current = currentPage.value;
      const maxVisible = 5; // 显示 5 个页码

      if (total <= maxVisible) {
        // 总页数少于 5 个，全部显示
        return Array.from({ length: total }, (_, i) => i + 1);
      }

      // 固定显示 5 个页码，当前页居中（如果可能）
      let start = Math.max(1, current - Math.floor(maxVisible / 2));
      let end = start + maxVisible - 1;

      // 调整边界
      if (end > total) {
        end = total;
        start = total - maxVisible + 1;
      }

      return Array.from({ length: end - start + 1 }, (_, i) => start + i);
    });

    // ========== 方法 ==========

    // 获取冻结列宽度
    const getFreezeWidths = (thElements) => {
      const leftCount = props.freeze?.left || 0;
      const rightCount = props.freeze?.right || 0;
      let leftWidth = 0;
      let rightWidth = 0;

      for (let i = 0; i < leftCount && i < thElements.length; i++) {
        leftWidth += thElements[i].offsetWidth;
      }
      for (let i = 0; i < rightCount && i < thElements.length; i++) {
        rightWidth += thElements[thElements.length - 1 - i].offsetWidth;
      }
      return { leftWidth, rightWidth };
    };

    // 滚动确保列可见
    const scrollColumnIntoView = (colIndex) => {
      if (!tableContainer.value) return;

      const thElements = tableContainer.value.querySelectorAll('thead th');
      const targetTh = thElements[colIndex];
      if (!targetTh) return;

      const containerWidth = tableContainer.value.clientWidth;
      const scrollLeft = tableContainer.value.scrollLeft;
      const colLeft = targetTh.offsetLeft;
      const colRight = colLeft + targetTh.offsetWidth;

      const { leftWidth: freezeLeftWidth, rightWidth: freezeRightWidth } = getFreezeWidths(thElements);

      // 实际可视区域（扣除冻结列）
      const actualViewLeft = scrollLeft + freezeLeftWidth;
      const actualViewRight = scrollLeft + containerWidth - freezeRightWidth;

      // 计算左右遮挡量
      const leftOverlap = actualViewLeft - colLeft;
      const rightOverlap = colRight - actualViewRight;

      // 优先处理遮挡更严重的一侧
      if (leftOverlap > 10 && leftOverlap >= rightOverlap) {
        // 向左滚动，让列紧贴左侧冻结列
        const newScrollLeft = Math.max(0, colLeft - freezeLeftWidth);
        tableContainer.value.scrollLeft = newScrollLeft;
      } else if (rightOverlap > 10 && rightOverlap > leftOverlap) {
        // 向右滚动，让列紧贴右侧冻结列
        tableContainer.value.scrollLeft = colRight - containerWidth + freezeRightWidth + 20;
      }
    };

    // 排序处理 - 按点击顺序形成优先级
    const handleSort = (col) => {
      const field = col.field;
      const existingIndex = multiSort.value.findIndex(s => s.field === field);
      const colIndex = displayCols.value.findIndex(c => c.field === field);

      if (existingIndex >= 0) {
        const existing = multiSort.value[existingIndex];
        if (existing.order === 'desc') {
          existing.order = 'asc';
        } else {
          multiSort.value.splice(existingIndex, 1);
          multiSort.value.forEach((item, idx) => {
            item.priority = idx + 1;
          });
        }
      } else {
        multiSort.value.push({ field, order: 'desc', priority: multiSort.value.length + 1 });
      }

      currentPage.value = 1;

      // 自动滚动确保排序图标可见
      nextTick(() => scrollColumnIntoView(colIndex));
    };

    // 获取排序指示器（返回对象供模板使用）
    const getSortIndicator = (col) => {
      const multiIndex = multiSort.value.findIndex(s => s.field === col.field);
      if (multiIndex >= 0) {
        const sort = multiSort.value[multiIndex];
        return {
          show: true,
          icon: sort.order === 'asc' ? '▲' : '▼',
          priority: sort.priority
        };
      }
      return { show: false };
    };

    // 判断是否冻结列
    const isFreezeCol = (index) => {
      if (!hasFreeze.value) return false;
      const left = props.freeze.left || 0;
      const right = props.freeze.right || 0;
      const total = displayCols.value.length;

      return index < left || index >= total - right;
    };

    // 获取冻结列的 CSS 类
    const getFreezeClass = (index) => {
      if (!isFreezeCol(index)) return '';
      const left = props.freeze.left || 0;
      return index < left ? 'freeze-col freeze-left' : 'freeze-col freeze-right';
    };

    // 应用冻结列样式
    const applyFreezeStyles = () => {
      if (!hasFreeze.value || !tableContainer.value) return;

      const table = tableContainer.value.querySelector('table');
      if (!table) return;

      const headerCells = table.querySelectorAll('thead th');
      const rows = table.querySelectorAll('tbody tr');

      // 获取实际列宽
      const colWidths = Array.from(headerCells).map(th => th.offsetWidth);

      // 左侧冻结
      let leftOffset = 0;
      const leftCount = props.freeze.left || 0;

      for (let i = 0; i < leftCount && i < headerCells.length; i++) {
        headerCells[i].style.left = `${leftOffset}px`;
        rows.forEach(row => {
          if (row.cells[i]) row.cells[i].style.left = `${leftOffset}px`;
        });
        leftOffset += colWidths[i];
      }

      // 右侧冻结
      let rightOffset = 0;
      const rightCount = props.freeze.right || 0;
      const totalCols = colWidths.length;

      for (let i = 0; i < rightCount && i < totalCols; i++) {
        const colIndex = totalCols - 1 - i;
        headerCells[colIndex].style.right = `${rightOffset}px`;
        rows.forEach(row => {
          if (row.cells[colIndex]) row.cells[colIndex].style.right = `${rightOffset}px`;
        });
        rightOffset += colWidths[colIndex];
      }
    };

    // 重置分页到第一页
    const resetPage = () => {
      currentPage.value = 1;
    };

    // 处理分页变化
    const handlePageChange = (page) => {
      if (page < 1 || page > totalPages.value) return;
      currentPage.value = page;
    };

    // 处理每页条数变化
    const handlePageSizeChange = (size) => {
      pageSize.value = size;
      currentPage.value = 1;
    };

    // ========== 滚动按钮显示/隐藏逻辑 ==========
    // 状态机：waiting(等待显示) → visible(显示) → hiding(等待消失) → waiting
    // 优点：状态转换时自动清除所有计时器，显示和隐藏不会同时进行
    //
    // 配置常量
    const showScrollButtons = ref(false);      // 按钮显示状态
    const scrollButtonPos = ref({ x: 0, y: 0 }); // 按钮位置（fixed 定位），显示后不再改变
    let buttonTimer = null;                    // 单一计时器（显示或隐藏共用）
    let btnState = 'waiting';                  // 按钮状态：waiting | visible | hiding
    let isOnButton = false;                   // 鼠标是否在按钮上
    let lastMousePos = { x: 0, y: 0 };         // 上次鼠标位置
    const HOVER_DELAY = 1000;                  // 悬停显示延迟（毫秒）
    const HIDE_DELAY = 600;                   // 隐藏延迟（毫秒）
    const MOVE_THRESHOLD = 25;                 // 移动阈值（像素）

    // 显示按钮
    const showButtons = () => {
      if (btnState !== 'waiting') return;
      btnState = 'visible';
      showScrollButtons.value = true;
      if (buttonTimer) clearTimeout(buttonTimer);
      buttonTimer = null;
    };

    // 保持显示（取消隐藏，恢复显示）
    const keepButtons = () => {
      if (btnState !== 'hiding') return;
      btnState = 'visible';
      if (buttonTimer) clearTimeout(buttonTimer);
      buttonTimer = null;
    };

    // 隐藏按钮（延迟隐藏）
    const hideButtons = () => {
      if (btnState !== 'visible') return;
      btnState = 'hiding';
      if (buttonTimer) clearTimeout(buttonTimer);
      buttonTimer = setTimeout(() => {
        showScrollButtons.value = false;
        btnState = 'waiting';
        buttonTimer = null;
      }, HIDE_DELAY);
    };

    // 检查鼠标是否在 tbody 区域
    const isMouseOnBody = (e) => {
      // 直接使用 e.target 判断是否在 tbody 内
      const target = e.target;
      if (!target) return false;
      // 检查 target 本身或其父元素是否是 tbody 或 tbody 的子元素
      const tbody = target.closest('tbody');
      return tbody !== null;
    };

    // 鼠标进入表格：启动显示计时
    const handleTableMouseEnter = (e) => {
      if (!props.scrollButton || !hasHorizontalScroll.value) return;
      // 只有在 tbody 区域才触发显示
      if (!isMouseOnBody(e)) return;
      lastMousePos = { x: e.clientX, y: e.clientY };
      if (btnState === 'waiting') {
        scrollButtonPos.value = { x: e.clientX, y: e.clientY };
        if (buttonTimer) clearTimeout(buttonTimer);
        buttonTimer = setTimeout(showButtons, HOVER_DELAY);
      }
    };

    // 鼠标在表格内移动
    const handleTableMouseMove = (e) => {
      if (!props.scrollButton) return;
      // 只有在 tbody 区域才触发显示
      if (!isMouseOnBody(e)) {
        if (btnState === 'waiting' && buttonTimer) {
          clearTimeout(buttonTimer);
          buttonTimer = null;
        }
        return;
      }
      // 计算移动距离
      const dx = e.clientX - lastMousePos.x;
      const dy = e.clientY - lastMousePos.y;
      const moved = Math.sqrt(dx * dx + dy * dy) > MOVE_THRESHOLD;

      // waiting 状态：更新位置，启动显示计时
      if (btnState === 'waiting') {
        scrollButtonPos.value = { x: e.clientX, y: e.clientY };
        if (buttonTimer) clearTimeout(buttonTimer);
        buttonTimer = setTimeout(showButtons, HOVER_DELAY);
      }
      // visible 状态：移动超过阈值且不在按钮上，触发隐藏
      if (btnState === 'visible' && moved && !isOnButton) {
        lastMousePos = { x: e.clientX, y: e.clientY };
        hideButtons();
      }
      // visible 状态：移动时更新位置
      if (btnState === 'visible' && moved) {
        lastMousePos = { x: e.clientX, y: e.clientY };
      }
    };

    // 鼠标离开表格：触发隐藏
    const handleTableMouseLeave = (e) => {
      // 检查鼠标是否移动到了按钮区域，如果是则不隐藏
      const buttonEl = document.querySelector('.ft-table-scroll-float');
      if (buttonEl && buttonEl.contains(e.relatedTarget)) {
        return;
      }
      if (btnState === 'visible') {
        hideButtons();
      }
    };

    // 滚轮滚动时检查鼠标是否离开 tbody
    const handleTableWheel = (e) => {
      if (!props.scrollButton || btnState !== 'visible') return;

      const tbody = tableContainer.value?.querySelector('tbody');
      if (!tbody) return;

      const rect = tbody.getBoundingClientRect();
      const isOutside =
        e.clientX < rect.left ||
        e.clientX > rect.right ||
        e.clientY < rect.top ||
        e.clientY > rect.bottom;

      if (isOutside) {
        hideButtons();
      }
    };

    // 鼠标进入按钮：保持显示
    const handleButtonMouseEnter = () => {
      isOnButton = true;
      keepButtons();
    };

    // 鼠标离开按钮：触发隐藏
    const handleButtonMouseLeave = () => {
      isOnButton = false;
      if (btnState === 'visible') {
        hideButtons();
      }
    };

    const scrollTable = (direction) => {
      if (!tableContainer.value) return;
      const scrollAmount = 120;
      if (direction === 'left') {
        tableContainer.value.scrollLeft -= scrollAmount;
      } else {
        tableContainer.value.scrollLeft += scrollAmount;
      }
    };

    // 格式化单元格值
    const formatValue = (value) => {
      if (value === null || value === undefined) return '';
      if (typeof value === 'number') {
        return Number.isInteger(value) ? value : value.toFixed(2);
      }
      return value;
    };

    // 获取单元格样式类
    const getCellClass = (value, col, colIndex, rowIndex) => {
      // 如果在热力图范围内，不应用正负颜色类（避免与热力图背景色冲突）
      if (isHeatmapCell(col, colIndex, rowIndex)) {
        return '';
      }
      if (typeof value === 'number') {
        return value >= 0 ? 'positive' : 'negative';
      }
      return '';
    };

    // 颜色插值计算（支持双色/三色）
    const interpolateColor = (colors, ratio) => {
      if (!colors || colors.length < 2) return '#ffffff';

      if (colors.length === 2) {
        return interpolateTwoColors(colors[0], colors[1], ratio);
      }

      if (ratio <= 0.5) {
        return interpolateTwoColors(colors[0], colors[1], ratio * 2);
      } else {
        return interpolateTwoColors(colors[1], colors[2], (ratio - 0.5) * 2);
      }
    };

    // 双色插值
    const interpolateTwoColors = (color1, color2, ratio) => {
      // 处理 3 位简写颜色 (#fff → #ffffff)
      const expandHex = (hex) => {
        hex = hex.replace('#', '');
        if (hex.length === 3) {
          return hex.split('').map(c => c + c).join('');
        }
        return hex;
      };

      const hex1 = expandHex(color1);
      const hex2 = expandHex(color2);

      const r1 = parseInt(hex1.substring(0, 2), 16);
      const g1 = parseInt(hex1.substring(2, 4), 16);
      const b1 = parseInt(hex1.substring(4, 6), 16);

      const r2 = parseInt(hex2.substring(0, 2), 16);
      const g2 = parseInt(hex2.substring(2, 4), 16);
      const b2 = parseInt(hex2.substring(4, 6), 16);

      const r = Math.round(r1 + (r2 - r1) * ratio);
      const g = Math.round(g1 + (g2 - g1) * ratio);
      const b = Math.round(b1 + (b2 - b1) * ratio);

      return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
    };

    // 判断单元格是否在热力图范围内
    const isHeatmapCell = (col, colIndex, rowIndex) => {
      if (!hasHeatmap.value) return false;

      if (isFreezeCol(colIndex)) return false;

      if (col.heatmap) return true;

      const hasGlobalHeatmap = props.heatmap && typeof props.heatmap === 'object';
      if (!hasGlobalHeatmap) return false;

      const config = heatmapConfig.value;
      if (!config) return false;

      const cols = displayCols.value;

      // 将分页索引转换为全局索引
      const globalRowIndex = pageConfig.value 
        ? (currentPage.value - 1) * pageSize.value + rowIndex 
        : rowIndex;

      const excludeRowsSet = new Set(
        (config.excludeRows || []).map(idx => idx < 0 ? props.data.length + idx : idx)
      );

      if (excludeRowsSet.has(globalRowIndex)) return false;

      if (config.columns && config.columns.length > 0) {
        return config.columns.includes(col.field);
      }

      const totalCols = cols.length;
      const start = config.start > 0 ? config.start - 1 : totalCols + config.start;
      const end = config.end > 0 ? config.end - 1 : totalCols + config.end;
      const excludeSet = new Set((config.exclude || []).map(v => {
        return v > 0 ? v - 1 : totalCols + v;
      }));

      if (colIndex < start || colIndex > end) return false;
      if (excludeSet.has(colIndex)) return false;

      return true;
    };

    // 获取单元格热力图样式
    const getHeatmapStyle = (value, col, colIndex, rowIndex) => {
      if (!hasHeatmap.value) return {};
      if (!isHeatmapCell(col, colIndex, rowIndex)) return {};
      if (typeof value !== 'number' || isNaN(value)) return {};

      const config = heatmapConfig.value;
      const ranges = heatmapRanges.value;

      let colors = config?.colors || ['#2196f3', '#fff', '#f44336'];
      let rangeKey = col.field;

      const colHeatmap = parseColHeatmap(col);
      if (colHeatmap) {
        if (colHeatmap.colors) {
          colors = colHeatmap.colors;
        }
        rangeKey = colHeatmap.group;
      }

      const range = ranges[rangeKey];

      if (!range) return {};

      const { min, max } = range;
      if (min === max) {
        const midColor = colors.length === 3 ? colors[1] : colors[1];
        return {
          'background-color': midColor,
          '--heatmap-bg': midColor
        };
      }

      const ratio = (value - min) / (max - min);
      const bgColor = interpolateColor(colors, ratio);

      return {
        'background-color': bgColor,
        '--heatmap-bg': bgColor
      };
    };

    // ========== 生命周期 ==========

    onMounted(() => {
      injectTableStyles();

      // 监听全局滚轮事件，用于检测鼠标是否离开表格
      if (props.scrollButton) {
        document.addEventListener('wheel', handleTableWheel, { passive: true });
      }

      if (hasFreeze.value) {
        nextTick(() => {
          applyFreezeStyles();

          resizeObserver.value = new ResizeObserver(() => {
            applyFreezeStyles();
          });

          if (tableContainer.value) {
            resizeObserver.value.observe(tableContainer.value);
          }
        });
      }
    });

    onUnmounted(() => {
      if (resizeObserver.value) {
        resizeObserver.value.disconnect();
      }
      if (props.scrollButton) {
        document.removeEventListener('wheel', handleTableWheel);
      }
    });

    // 监听数据变化
    watch(() => props.data, () => {
      // 如果 resetPage 为 true，数据变化时重置到第一页
      if (props.resetPage) {
        resetPage();
      } else {
        // 即使不重置，也要确保当前页不超过总页数
        nextTick(() => {
          if (currentPage.value > totalPages.value) {
            currentPage.value = totalPages.value;
          }
        });
      }
      if (hasFreeze.value) {
        nextTick(() => {
          applyFreezeStyles();
        });
      }
    }, { deep: true });

    // 监听列变化
    watch(() => displayCols.value, () => {
      if (hasFreeze.value) {
        nextTick(() => {
          applyFreezeStyles();
        });
      }
    });

    // 监听分页变化
    watch(currentPage, () => {
      if (hasFreeze.value) {
        nextTick(() => {
          applyFreezeStyles();
        });
      }
    });

    // 监听 freeze 配置变化
    watch(() => props.freeze, () => {
      if (hasFreeze.value) {
        nextTick(() => {
          applyFreezeStyles();
        });
      }
    }, { deep: true });

    // ========== 注入 CSS ==========
    // 仅注入冻结列必需的最简样式
    // 其他样式（表格基础、分页等）请引入 ft-table.css
    const injectTableStyles = () => {
      const styleId = 'ft-table-freeze-core';
      if (document.getElementById(styleId)) return;

      const style = document.createElement('style');
      style.id = styleId;
      style.textContent = `
        /* 冻结列核心样式 - 组件必需 */
        .ft-table-freeze {
          overflow-x: auto;
          position: relative;
        }
        .ft-table-freeze .freeze-col {
          position: sticky;
        }
        .ft-table-freeze thead th {
          position: sticky;
          top: 0;
          z-index: 10;
        }
        .ft-table-freeze thead .freeze-col {
          z-index: 100;
        }
        .ft-table-freeze tbody .freeze-col {
          z-index: 50;
        }
        .ft-table-freeze .ft-table {
          border-collapse: separate;
          border-spacing: 0;
        }
        .ft-table-freeze .freeze-left {
          box-shadow: 2px 0 4px rgba(0, 0, 0, 0.1);
        }
        .ft-table-freeze .freeze-right {
          box-shadow: -2px 0 4px rgba(0, 0, 0, 0.1);
        }
        /* 排序图标基础样式 - 保证无外部样式时依然可辨识 */
        span.sort-icon {
          font-size: 0.8em;
        }
        span.sort-priority {
          font-size: 0.7em;
        }
        /* 热力图单元格样式 - 确保背景色优先级最高 */
        .ft-table td.heatmap-cell {
          background-color: var(--heatmap-bg) !important;
        }
        /* 悬停滚动按钮 */
        .ft-table-wrapper {
          position: relative;
        }
        .ft-table-scroll-float {
          position: fixed;
          z-index: 1000;
          display: flex;
          gap: 8px;
          pointer-events: none;
          opacity: 0;
          transition: opacity 0.3s;
        }
        .ft-table-scroll-float.visible {
          opacity: 1;
        }
        .ft-table-scroll-float button {
          pointer-events: auto;
          width: 36px;
          height: 36px;
          border-radius: 50%;
          background: rgba(64, 64, 64, 0.85);
          color: white;
          border: none;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 14px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .ft-table-scroll-float button:hover {
          background: rgba(0, 0, 0, 0.9);
        }
      `;
      document.head.appendChild(style);
    };

    return {
      tableContainer,
      displayCols,
      paginatedData,
      currentPage,
      totalPages,
      pageSize,
      pageConfig,
      hasFreeze,
      hasHorizontalScroll,
      hasHeatmap,
      pageSizeOptions,
      visiblePages,
      showScrollButtons,
      scrollButtonPos,
      scrollButton: props.scrollButton,
      handleButtonMouseEnter,
      handleButtonMouseLeave,
      handleTableMouseMove,
      handleTableMouseEnter,
      handleTableMouseLeave,
      handleTableWheel,
      scrollTable,
      handleSort,
      getSortIndicator,
      isFreezeCol,
      getFreezeClass,
      handlePageChange,
      handlePageSizeChange,
      formatValue,
      getCellClass,
      isHeatmapCell,
      getHeatmapStyle,
      hasSlot,
      resetPage
    };
  },

  template: `
    <div class="ft-table-wrapper">
      <div 
        ref="tableContainer"
        class="ft-table-container"
        :class="{ 'ft-table-freeze': hasFreeze }"
        @mouseenter="handleTableMouseEnter"
        @mouseleave="handleTableMouseLeave"
        @mousemove="handleTableMouseMove"
      >
        <table class="ft-table">
          <thead>
            <tr>
              <th 
                v-for="(col, index) in displayCols" 
                :key="col.field"
                :class="[getFreezeClass(index), { 'no-sort': col.sort === false }]"
                @click="col.sort !== false && handleSort(col)"
              >
                {{ col.title }}
                <template v-if="col.sort !== false" v-for="indicator in [getSortIndicator(col)]" :key="col.field + '-indicator'">
                  <span v-if="indicator.show" class="sort-indicator">
                    <span class="sort-priority">{{ indicator.priority }}</span>
                    <span class="sort-icon">{{ indicator.icon }}</span>
                  </span>
                </template>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, rowIndex) in paginatedData" :key="rowIndex">
              <td 
                v-for="(col, colIndex) in displayCols" 
                :key="col.field"
                :class="[getFreezeClass(colIndex), getCellClass(row[col.field], col, colIndex, rowIndex), { 'heatmap-cell': isHeatmapCell(col, colIndex, rowIndex) }]"
                :style="getHeatmapStyle(row[col.field], col, colIndex, rowIndex)"
              >
                <!-- 有插槽：调用插槽渲染 -->
                <template v-if="col.slot && $slots[col.slot]">
                  <slot :name="col.slot" :row="row" :value="row[col.field]" :index="rowIndex" />
                </template>
                <!-- 无插槽：默认渲染 -->
                <template v-else>
                  {{ formatValue(row[col.field]) }}
                </template>
              </td>
            </tr>
          </tbody>
        </table>
        
        <!-- 空数据提示 -->
        <div v-if="paginatedData.length === 0" class="ft-table-empty">
          {{ emptyText }}
        </div>
      </div>
      
      <!-- 悬停3秒后显示的滚动按钮（仅当有横向滚动时显示） -->
      <div 
        v-if="hasHorizontalScroll && scrollButton"
        class="ft-table-scroll-float" 
        :class="{ visible: showScrollButtons }"
        :style="{ left: scrollButtonPos.x + 'px', top: scrollButtonPos.y + 'px' }"
        @mouseenter="handleButtonMouseEnter"
        @mouseleave="handleButtonMouseLeave"
      >
        <button @click="scrollTable('left')">◀</button>
        <button @click="scrollTable('right')">▶</button>
      </div>
      
      <!-- 分页：数据量超过每页条数时显示 -->
      <div v-if="pageConfig !== false && data.length > pageSize" class="ft-table-pagination">
        <!-- 首页 -->
        <button 
          @click="handlePageChange(1)"
          :disabled="currentPage <= 1"
          class="page-btn"
        >
          首页
        </button>
        
        <!-- 上一页 -->
        <button 
          @click="handlePageChange(currentPage - 1)"
          :disabled="currentPage <= 1"
          class="page-btn"
        >
          上一页
        </button>
        
        <!-- 页码 -->
        <button 
          v-for="page in visiblePages" 
          :key="page"
          @click="handlePageChange(page)"
          :class="['page-btn', { active: page === currentPage }]"
        >
          {{ page }}
        </button>
        
        <!-- 下一页 -->
        <button 
          @click="handlePageChange(currentPage + 1)"
          :disabled="currentPage >= totalPages"
          class="page-btn"
        >
          下一页
        </button>
        
        <!-- 末页 -->
        <button 
          @click="handlePageChange(totalPages)"
          :disabled="currentPage >= totalPages"
          class="page-btn"
        >
          末页
        </button>
        
        <!-- 分隔 -->
        <span class="page-info">
          第 {{ currentPage }} / {{ totalPages }} 页
        </span>
        
        <!-- 每页条数 -->
        <select v-model="pageSize" @change="handlePageSizeChange(pageSize)">
          <option v-for="size in pageSizeOptions" :key="size" :value="size">
            {{ size }}条/页
          </option>
        </select>
      </div>
    </div>
  `
};

// 全局暴露（CDN 使用方式）
if (typeof window !== 'undefined') {
  window.FtTable = FtTable;
  // 保留旧名称兼容
  window.VueTable = FtTable;
}

// ES Module 导出
if (typeof exports !== 'undefined') {
  exports.FtTable = FtTable;
  exports.VueTable = FtTable;
}
