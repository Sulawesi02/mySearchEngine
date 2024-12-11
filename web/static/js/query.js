let results = []; // 存储所有查询结果
let currentPage = 1; // 当前页码
const itemsPerPage = 10; // 每页显示的结果数

// 切换高级搜索窗口的显示/隐藏
function toggleAdvancedquery() {
    var popup = document.getElementById('advanced-query-popup');
    if (popup.style.display === "none" || popup.style.display === "") {
      popup.style.display = "block";
    } else {
      popup.style.display = "none";
    }
}

// 普通搜索
function commonQuery(queryInput, queryResults){
    
    const queryStr = queryInput.value.trim();

    if (!queryStr) {
        alert('请输入查询内容！');
        return;
    }

    fetch('/common_query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query_str: queryStr
        })
    })
    .then(response => response.json())
    .then(data => {
        currentPage = 1;
        displayQueryResults(data, queryResults);
    })
    .catch(error => {
        console.error('查询过程中出现错误:', error);
        queryResults.innerHTML = '<p>查询过程中出现错误，请稍后重试。</p>';
        queryResults.style.display = 'block';
    });
}

// 展示查询结果
function displayQueryResults(data, queryResults) {
    if (data.success) {
        // 存储所有查询结果
        results = data.results;

        // 渲染第一页的数据
        renderPage(currentPage, queryResults);

        // 更新分页控件
        updatePaginationControls();

        // 如果有查询结果，显示分页控件
        if (results.length > 0) {
            const paginationControls = document.getElementById('pagination-controls');
            paginationControls.style.display = 'flex';
        }

        // 显示搜索结果区域
        queryResults.style.display = 'block';
    } else {
        queryResults.innerHTML = '<p>查询失败，请稍后重试。</p>';
        queryResults.style.display = 'block';

        // 如果查询失败，隐藏分页控件
        const paginationControls = document.getElementById('pagination-controls');
        paginationControls.style.display = 'none';
    }
}

// 渲染当前页面的数据
function renderPage(page, queryResults) {
    // 清空之前的搜索结果
    queryResults.innerHTML = '';

    // 计算当前页面的起始和结束索引
    const start = (page - 1) * itemsPerPage;
    const end = start + itemsPerPage;

    // 获取当前页面的数据
    const pageResults = results.slice(start, end);

    // 遍历并展示当前页面的查询结果
    if (pageResults.length === 0) {
        queryResults.innerHTML = '<p>未找到相关结果。</p>';
    } else {
        pageResults.forEach((result, index) => {
            const article = document.createElement('article');
            article.classList.add('query-result');

            const h3 = document.createElement('h3');
            h3.textContent = `${start + index + 1}. ${result.title}`;
            h3.addEventListener('click', function () {
                openLocalXmlAsHtml(result.doc_id);
            });

            const p = document.createElement('p');
            p.textContent = result.content;

            const a = document.createElement('a');
            a.textContent = result.url;
            a.href = result.url;
            a.target = '_blank'; // 在新标签页中打开链接

            const snapShotLink = document.createElement('a');
            snapShotLink.textContent = '查看快照';
            snapShotLink.href = result.snap_shot_url;
            snapShotLink.target = '_blank';  // 在新窗口中打开快照
            // 为 snapShotLink 绑定点击事件处理器
            snapShotLink.addEventListener('click', function () {
                openSnapshot(result.snap_shot_url);
            });


            article.appendChild(h3);
            article.appendChild(p);
            article.appendChild(a);
            article.appendChild(snapShotLink);

            queryResults.appendChild(article);
        });
    }
}

// 构建后端生成 HTML 的 URL，并在新标签页中打开
function openLocalXmlAsHtml(doc_id) {
    fetch(`/get_html/${doc_id}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('获取HTML内容失败');
            }
            return response.text();
        })
        .then(html => {
            // 创建新标签页并写入 HTML 内容
            const newWindow = window.open('', '_blank');
            if (newWindow) {
                newWindow.document.write(html);
                newWindow.document.close();  // 确保页面加载完毕
            } else {
                alert('无法打开新标签页，请检查浏览器设置');
            }
        })
        .catch(error => {
            console.error('获取HTML内容出现错误:', error);
            alert('获取HTML内容时出错，请稍后再试');
        });
}

// 更新分页控件的状态
function updatePaginationControls() {
    const totalPages = Math.ceil(results.length / itemsPerPage);
    const prevPageButton = document.getElementById('prev-page');
    const nextPageButton = document.getElementById('next-page');
    const pageNumbersContainer = document.getElementById('page-numbers');

    // 清空之前的页码链接
    pageNumbersContainer.innerHTML = '';

    // 禁用/启用“上一页”按钮
    if (currentPage === 1) {
        prevPageButton.disabled = true;
    } else {
        prevPageButton.disabled = false;
    }

    // 禁用/启用“下一页”按钮
    if (currentPage === totalPages || totalPages === 0) {
        nextPageButton.disabled = true;
    } else {
        nextPageButton.disabled = false;
    }

    // 如果总页数为0，直接返回并隐藏分页控件
    if (totalPages === 0) {
        const paginationControls = document.getElementById('pagination-controls');
        paginationControls.style.display = 'none';
        return;
    }

    // 定义显示的页码范围
    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(totalPages, currentPage + 2);

    // 如果起始页码大于1，则显示“...”表示前面还有更多页码
    if (startPage > 1) {
        const ellipsisBefore = document.createElement('span');
        ellipsisBefore.textContent = '...';
        pageNumbersContainer.appendChild(ellipsisBefore);
    }

    // 动态生成页码链接
    for (let i = startPage; i <= endPage; i++) {
        const pageNumberLink = document.createElement('a');
        pageNumberLink.href = '#';
        pageNumberLink.textContent = i;
        pageNumberLink.classList.add('page-number-link');

        // 如果是当前页，添加 active 类
        if (i === currentPage) {
            pageNumberLink.classList.add('active');
        }

        // 为页码链接添加点击事件
        pageNumberLink.addEventListener('click', function (event) {
            event.preventDefault(); // 防止默认行为
            currentPage = i;
            renderPage(currentPage, document.getElementById('query-results'));
            updatePaginationControls();
        });

        pageNumbersContainer.appendChild(pageNumberLink);
    }

    // 如果结束页码小于总页数，则显示“...”表示后面还有更多页码
    if (endPage < totalPages) {
        const ellipsisAfter = document.createElement('span');
        ellipsisAfter.textContent = '...';
        pageNumbersContainer.appendChild(ellipsisAfter);
    }

    // 如果总页数超过5页，始终显示第一页和最后一页
    if (totalPages > 5) {
        if (startPage > 1) {
            const firstPageLink = document.createElement('a');
            firstPageLink.href = '#';
            firstPageLink.textContent = '1';
            firstPageLink.classList.add('page-number-link');

            firstPageLink.addEventListener('click', function (event) {
                event.preventDefault();
                currentPage = 1;
                renderPage(currentPage, document.getElementById('query-results'));
                updatePaginationControls();
            });

            pageNumbersContainer.insertBefore(firstPageLink, pageNumbersContainer.firstChild);
        }

        if (endPage < totalPages) {
            const lastPageLink = document.createElement('a');
            lastPageLink.href = '#';
            lastPageLink.textContent = totalPages;
            lastPageLink.classList.add('page-number-link');

            lastPageLink.addEventListener('click', function (event) {
                event.preventDefault();
                currentPage = totalPages;
                renderPage(currentPage, document.getElementById('query-results'));
                updatePaginationControls();
            });

            pageNumbersContainer.appendChild(lastPageLink);
        }
    }

    // 确保分页控件在有结果时显示
    const paginationControls = document.getElementById('pagination-controls');
    if (totalPages > 0) {
        paginationControls.style.display = 'flex';
    }
}


// 在新标签页中打开快照
function openSnapshot(screenshotUrl) {

    // 构建新页面的HTML内容
    const newPageHtml = `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>View Screenshot</title>
                <style>
                    body, html {
                        margin: 0;
                        padding: 0;
                        height: 100%;
                        overflow: hidden; /* 防止页面滚动 */
                    }
                   .screenshot-container {
                        width: 100%;
                        height: 100%;
                        justify-content: center;
                        align-items: center;
                        overflow: auto; /* 允许图片超出部分可以滚动 */
                    }
                   .screenshot-container img {
                        max-width: 100%;
                        height: auto;
                        object-fit: contain;
                    }
                </style>
            </head>
            <body>
                <div class="screenshot-container">
                    <img src="${screenshotUrl}" alt="Page Snapshot">
                </div>
            </body>
            </html>
        `;

    // 创建一个Blob对象，将HTML字符串转换为可下载的内容
    const blob = new Blob([newPageHtml], { type: 'text/html' });
    const url = URL.createObjectURL(blob);

    // 在新标签页中打开快照
    window.open(url, '_blank');
}

// 删除查询日志
function deleteHistory(historyId) {
    fetch(`/delete_query_history/${historyId}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('删除成功');
        } else {
            console.error('删除失败:', data.message);
        }
    })
    .catch(error => {
        console.error('删除记录失败:', error);
    });
}

document.addEventListener('DOMContentLoaded', function () {

    // 初始化时隐藏分页控件
    const paginationControls = document.getElementById('pagination-controls');
    paginationControls.style.display = 'none';

    // 普通搜索
    const queryInput = document.getElementById('query-input');
    const queryButton = document.getElementById('query-button');
    const queryResults = document.getElementById('query-results');

    queryButton.addEventListener('click', function (event) {
        queryHistory.style.display = 'none';
        event.preventDefault(); // 防止表单默认提交
        commonQuery(queryInput, queryResults);
    });

    // 高级搜索
    const advancedqueryButton = document.getElementById('advanced-query-button');
    const advancedqueryInput = document.getElementById('advanced-query-input');
    const inStationQueryUrlInput = document.getElementById('in_station_query_url_input');
    const queryOption = document.getElementsByName('query_type');

    advancedqueryButton.addEventListener('click', function (event) {
        event.preventDefault(); // 防止表单默认提交
        const queryStr = advancedqueryInput.value.trim();
        const queryUrl = inStationQueryUrlInput.value.trim();

        if (!queryStr) {
            alert('请输入查询内容！');
            return;
        }

        // 获取选中的查询类型
        let queryType = '';
        queryOption.forEach(option => {
            if (option.checked) {
                queryType = option.value;
            }
        });

        if (!queryType) {
            alert('请选择查询类型！');
            return;
        }

        let url = '';
        if (queryType === 'instation') {
            url = '/instation_query';
        } else if (queryType === 'document') {
            url = '/document_query';
        } else if (queryType === 'phrase') {
            url = '/phrase_query';
        } else if (queryType === 'wildcard') {
            url = '/wildcard_query';
        }

        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query_str: queryStr,
                query_url: queryUrl
            })
        })
      .then(response => response.json())
      .then(data => {
            displayQueryResults(data, queryResults);
        })
      .catch(error => {
            console.error('查询过程中出现错误:', error);
            queryResults.innerHTML = '<p>查询过程中出现错误，请稍后重试。</p>';
            queryResults.style.display = 'block';
        });
    });

    const prevPage = document.getElementById('prev-page');
    const nextPage = document.getElementById('next-page');

    // 点击“上一页”按钮
    prevPage.addEventListener('click', function () {
        if (currentPage > 1) {
            currentPage--;
            renderPage(currentPage, document.getElementById('query-results'));
            updatePaginationControls();
        }
    });

    // 点击“下一页”按钮
    nextPage.addEventListener('click', function () {
        const totalPages = Math.ceil(results.length / itemsPerPage);
        if (currentPage < totalPages) {
            currentPage++;
            renderPage(currentPage, document.getElementById('query-results'));
            updatePaginationControls();
        }
    });


    // 查询日志
    const queryHistory = document.getElementById('query-history');

    // 用户点击输入框显示历史记录
    queryInput.addEventListener('click', function (event) {
        // 如果输入框没有文本
        if (queryInput.value === '') {
        fetch('/get_query_history')
          .then(response => response.json())
          .then(data => {
            if (data.success) {
              const history = data.history;

              // 清空之前的查询日志
              queryHistory.innerHTML = '';

              // 动态生成查询日志项
              history.forEach(item => {
                const historyItem = document.createElement('div');
                historyItem.classList.add('query-history-item');

                // 为查询日志项添加点击事件
                historyItem.addEventListener('click', function () {
                  queryInput.value = item.query_str;
                  queryHistory.style.display = 'none';  // 点击后隐藏查询日志
                  commonQuery(queryInput, queryResults);
                });

                // 查询日志项文本
                const historyItemText = document.createElement('span');
                historyItemText.textContent = item.query_str;
                historyItem.appendChild(historyItemText);

                // 查询日志项删除按钮
                const historyDeleteButton = document.createElement('button');
                historyDeleteButton.classList.add('delete-button');
                historyDeleteButton.textContent = '删除';
                
                // 为日志项删除按钮添加点击事件
                historyDeleteButton.addEventListener('click', function (event) {
                  event.stopPropagation(); // 阻止点击冒泡到 historyItem
                  deleteHistory(item.id);// 删除记录
                  historyItem.remove();
                });
                historyItem.appendChild(historyDeleteButton);

                queryHistory.appendChild(historyItem);
              });

              queryHistory.style.display = 'block';
            } else {
              alert(data.message);
            }
          })
          .catch(error => {
            console.error('获取查询日志失败:', error);
          });
        }
    });
    
    // 用户输入文本，隐藏查询日志
    queryInput.addEventListener('input', function (event) {
        queryHistory.style.display = 'none';
    });

    // 用户点击页面其他区域，隐藏查询日志
    document.addEventListener('click', function (event) {
      if (!event.target.closest('.query-bar')) {
        queryHistory.style.display = 'none';
      }
    });

});