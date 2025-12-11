// 全局变量存储下载URL，避免被过早释放
let downloadUrls = [];

// 清理旧的下载URL
function cleanupOldDownloadUrls() {
    downloadUrls.forEach(url => {
        try {
            URL.revokeObjectURL(url);
        } catch (e) {
            // 忽略错误
        }
    });
    downloadUrls = [];
}

// 在页面卸载时清理所有URL
window.addEventListener('beforeunload', function() {
    cleanupOldDownloadUrls();
});

document.getElementById('sequenceForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const species = document.getElementById('speciesSelect').value;
    const sequence = document.getElementById('sequenceInput').value;
    const fileInput = document.getElementById('sequenceFile');
    const file = fileInput.files[0];
    const resultDiv = document.getElementById('result');
    
    // 清理旧的下载URL
    cleanupOldDownloadUrls();
    
    // Show enhanced loading message
    resultDiv.innerHTML = `
        <div class="loading" style="font-size: 22px; text-align: center; padding: 30px; color: #666;">
            <div style="margin-bottom: 15px;">🔬 Processing DNA sequences...</div>
            <div style="font-size: 18px; color: #0056b3;">
                <em>EnDeep4mC</em> is analyzing your sequences with variable-length processing
            </div>
            <div style="font-size: 16px; color: #666; margin-top: 10px;">
                • Standardizing sequences to 41bp windows<br>
                • Filtering windows with cytosine (C) at center<br>
                • Generating ensemble predictions
            </div>
        </div>`;

    if (!sequence && !file) {
        resultDiv.innerHTML = '<p class="error" style="font-size: 20px; color: #c62828; background: #ffebee; padding: 1.5rem; border-radius: 6px; border-left: 4px solid #c62828; text-align: center;">Please enter a DNA sequence or upload a file.</p>';
        return;
    }

    const formData = new FormData();
    formData.append('species', species);
    if (sequence) formData.append('sequence', sequence);
    if (file) formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            resultDiv.innerHTML = `<p class="error" style="font-size: 20px; color: #c62828; background: #ffebee; padding: 1.5rem; border-radius: 6px; border-left: 4px solid #c62828; text-align: center;">Error: ${data.error}</p>`;
            return;
        }
        
        // Generate download file content
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const speciesName = species.replace('4mC_', '');
        
        // 创建下载内容 - 只包含最终结果
        let downloadContent = `# EnDeep4mC Prediction Results\n`;
        downloadContent += `# Generated: ${new Date().toLocaleString()}\n`;
        downloadContent += `# Species: ${speciesName}\n`;
        downloadContent += `# Variable-length processing enabled\n`;
        downloadContent += `# Total Sequences: ${data.summary?.total_original_sequences || 'N/A'}\n`;
        downloadContent += `# Total Final Predictions: ${data.summary?.total_final_predictions || 'N/A'}\n\n`;
        
        data.final_results.forEach((item, index) => {
            downloadContent += `>${item.seq_id} | Original: ${item.original_seq_id} | Length: ${item.original_length}bp\n`;
            downloadContent += `${item.sequence}\n`;
            downloadContent += `Probability: ${(item.probability * 100).toFixed(2)}%\n`;
            downloadContent += `Prediction: ${item.is_4mC_site ? '4mC Site' : 'Negative'}\n`;
            if (item.sequence_length === 'long' && item.is_aggregated) {
                downloadContent += `# Cytosine Positions Analyzed: ${item.cytosine_positions}\n`;
                downloadContent += `# 4mC Positions Found: ${item['4mC_positions']}\n`;
            }
            downloadContent += `\n`;
        });
        
        // 创建下载链接
        const blob = new Blob([downloadContent], {type: 'text/plain;charset=utf-8'});
        const url = URL.createObjectURL(blob);
        downloadUrls.push(url); // 存储URL以便后续清理
        
        const filename = `EnDeep4mC_${speciesName}_${timestamp}.txt`;
        
        // Build result HTML - 重新设计布局
        let resultHtml = `
            <div style="background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #eaeaea;">
                <h2 style="font-size: 26px; color: #0056b3; border-bottom: 2px solid #1e90ff; padding-bottom: 10px; margin-bottom: 20px; text-align: center;">
                    🧬 Final Prediction Results
                </h2>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-bottom: 20px;">
                    <div style="background: #f0f7ff; padding: 15px; border-radius: 6px; border-left: 4px solid #1e90ff;">
                        <div style="font-size: 18px; font-weight: bold; color: #0056b3; margin-bottom: 5px;">Species</div>
                        <div style="color: #333; font-weight: bold; font-size: 18px;">${speciesName}</div>
                    </div>
                    <div style="background: #f0f7ff; padding: 15px; border-radius: 6px; border-left: 4px solid #28a745;">
                        <div style="font-size: 18px; font-weight: bold; color: #0056b3; margin-bottom: 5px;">Processing Summary</div>
                        <div style="color: #333; font-size: 18px;">
                            ${data.summary?.total_original_sequences || 0} sequence(s)<br>
                            ${data.summary?.total_final_predictions || 0} final prediction(s)
                        </div>
                    </div>
                    <div style="background: #f0f7ff; padding: 15px; border-radius: 6px; border-left: 4px solid #ffc107;">
                        <div style="font-size: 18px; font-weight: bold; color: #0056b3; margin-bottom: 5px;">Sequence Types</div>
                        <div style="color: #333; font-size: 18px;">
                            ${data.summary?.long_sequences || 0} long<br>
                            ${data.summary?.short_sequences || 0} short/41bp
                        </div>
                    </div>
                </div>
                
                <!-- 下载区域 - 居中显示 -->
                <div style="text-align: center; background: #e8f4ff; padding: 20px; border-radius: 6px; margin-bottom: 25px;">
                    <div style="font-size: 20px; font-weight: bold; color: #0056b3; margin-bottom: 10px;">
                        Download Final Results
                    </div>
                    <div style="font-size: 18px; color: #333; margin-bottom: 15px;">
                        Complete report with ${data.final_results.length} final prediction(s)
                    </div>
                    <a href="${url}" download="${filename}" 
                       style="display: inline-block; background-color: #28a745; color: white; padding: 12px 25px; 
                       font-size: 20px; border-radius: 6px; text-decoration: none; font-weight: bold;
                       border-bottom: 3px solid #1e7e34; transition: all 0.3s;">
                        📥 Download Final Report
                    </a>
                </div>`;
        
        if (data.final_results.length === 0) {
            resultHtml += `
                <div style="background: #fff3cd; padding: 20px; border-radius: 6px; border: 1px solid #ffc107; text-align: center;">
                    <div style="font-size: 24px; color: #856404; margin-bottom: 10px;">⚠️ No Predictions Generated</div>
                    <div style="font-size: 18px; color: #856404;">
                        No valid 41bp windows with cytosine (C) at center position were found.
                        <br>Please ensure your sequences contain cytosine (C) bases.
                    </div>
                </div>`;
        } else {
            // 显示最终结果
            resultHtml += `
                <div style="margin-top: 20px;">
                    <div style="text-align: center; margin-bottom: 20px;">
                        <h3 style="font-size: 24px; color: #0056b3; margin-bottom: 5px;">
                            Final Prediction Results
                        </h3>
                        <div style="font-size: 18px; color: #666;">
                            ${data.final_results.length} final prediction(s)
                        </div>
                    </div>
                    
                    <div style="overflow-x: auto;">
                        <table style="width: 100%; border-collapse: collapse; border: 1px solid #ddd; border-radius: 6px; overflow: hidden;">
                            <thead style="background: #f5f5f5;">
                                <tr>
                                    <th style="padding: 15px; text-align: left; border-bottom: 2px solid #ddd; font-size: 18px; font-weight: bold; color: #333;">Sequence ID</th>
                                    <th style="padding: 15px; text-align: center; border-bottom: 2px solid #ddd; font-size: 18px; font-weight: bold; color: #333;">Length</th>
                                    <th style="padding: 15px; text-align: center; border-bottom: 2px solid #ddd; font-size: 18px; font-weight: bold; color: #333;">Probability</th>
                                    <th style="padding: 15px; text-align: center; border-bottom: 2px solid #ddd; font-size: 18px; font-weight: bold; color: #333;">Prediction</th>
                                    <th style="padding: 15px; text-align: center; border-bottom: 2px solid #ddd; font-size: 18px; font-weight: bold; color: #333;">Details</th>
                                </tr>
                            </thead>
                            <tbody>`;
            
            data.final_results.forEach((item, index) => {
                const probPercent = (item.probability * 100).toFixed(1);
                const isPositive = item.is_4mC_site;
                const lengthInfo = item.sequence_length === 'long' ? 
                                 `${item.original_length}bp (long)` : 
                                 item.sequence_length === '41bp' ? '41bp' : 
                                 `${item.original_length}bp (short)`;
                const lengthColor = item.sequence_length === 'long' ? '#ff9800' : 
                                  item.sequence_length === '41bp' ? '#2196f3' : '#9c27b0';
                
                // 长序列的额外信息
                let detailsHtml = '';
                if (item.sequence_length === 'long' && item.is_aggregated) {
                    detailsHtml = `
                        <div style="font-size: 14px; color: #666;">
                            Analyzed ${item.cytosine_positions} cytosine positions<br>
                            Found ${item['4mC_positions']} potential 4mC sites
                        </div>`;
                }
                
                resultHtml += `
                    <tr style="${index % 2 === 0 ? 'background: #fafafa;' : 'background: white;'}">
                        <td style="padding: 12px; border-bottom: 1px solid #eee; font-size: 18px;">
                            <div style="font-weight: 600; color: #2c3e50;">${item.seq_id}</div>
                            <div style="font-size: 16px; color: #666; margin-top: 2px;">${item.sequence}</div>
                        </td>
                        <td style="padding: 12px; border-bottom: 1px solid #eee; text-align: center;">
                            <span style="display: inline-block; padding: 6px 12px; background: ${lengthColor}15; color: ${lengthColor}; border-radius: 4px; font-size: 18px; font-weight: 600;">
                                ${lengthInfo}
                            </span>
                        </td>
                        <td style="padding: 12px; border-bottom: 1px solid #eee; text-align: center;">
                            <div style="display: inline-block; position: relative; width: 120px; height: 30px; background: #e0e0e0; border-radius: 15px; overflow: hidden;">
                                <div style="position: absolute; top: 0; left: 0; height: 100%; background: ${isPositive ? '#4CAF50' : '#f44336'}; width: ${probPercent}%;"></div>
                                <span style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-weight: 600; font-size: 18px; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
                                    ${probPercent}%
                                </span>
                            </div>
                        </td>
                        <td style="padding: 12px; border-bottom: 1px solid #eee; text-align: center;">
                            <span style="display: inline-block; padding: 8px 16px; border-radius: 15px; font-size: 18px; font-weight: 600; ${isPositive ? 'background: #e8f5e9; color: #2e7d32;' : 'background: #ffebee; color: #c62828;'}">
                                ${isPositive ? '4mC Site ✔️' : 'Negative ❌'}
                            </span>
                        </td>
                        <td style="padding: 12px; border-bottom: 1px solid #eee; text-align: center;">
                            ${detailsHtml}
                        </td>
                    </tr>`;
            });
            
            resultHtml += `
                            </tbody>
                        </table>
                    </div>`;
            
            // 如果有窗口详细信息，显示折叠部分
            if (data.window_details && data.window_details.length > 0) {
                const windowDetailId = 'windowDetails_' + Date.now();
                resultHtml += `
                    <div style="margin-top: 30px; border: 1px solid #ddd; border-radius: 6px; overflow: hidden;">
                        <div style="background: #f5f5f5; padding: 15px; cursor: pointer; display: flex; justify-content: space-between; align-items: center;" 
                             onclick="toggleWindowDetails('${windowDetailId}')">
                            <div>
                                <h4 style="font-size: 20px; color: #0056b3; margin: 0;">
                                    🔍 Detailed Window Predictions
                                </h4>
                                <div style="font-size: 16px; color: #666; margin-top: 5px;">
                                    ${data.window_details.length} window-level predictions (Click to expand/collapse)
                                </div>
                            </div>
                            <span id="${windowDetailId}_icon" style="font-size: 24px;">▶️</span>
                        </div>
                        <div id="${windowDetailId}" style="display: none; padding: 20px; background: white; max-height: 400px; overflow-y: auto;">
                            <div style="overflow-x: auto;">
                                <table style="width: 100%; border-collapse: collapse; font-size: 16px;">
                                    <thead style="background: #f8f9fa;">
                                        <tr>
                                            <th style="padding: 10px; border-bottom: 1px solid #ddd; text-align: left;">Window ID</th>
                                            <th style="padding: 10px; border-bottom: 1px solid #ddd; text-align: center;">Position</th>
                                            <th style="padding: 10px; border-bottom: 1px solid #ddd; text-align: center;">Probability</th>
                                            <th style="padding: 10px; border-bottom: 1px solid #ddd; text-align: center;">Prediction</th>
                                        </tr>
                                    </thead>
                                    <tbody>`;
                
                // 显示前50个窗口详细信息（避免页面太长）
                const displayWindowDetails = data.window_details.slice(0, 50);
                displayWindowDetails.forEach((item, index) => {
                    const probPercent = (item.probability * 100).toFixed(1);
                    const isPositive = item.is_4mC_site;
                    
                    resultHtml += `
                        <tr style="${index % 2 === 0 ? 'background: #fafafa;' : 'background: white;'}">
                            <td style="padding: 8px; border-bottom: 1px solid #eee; font-size: 14px;">
                                ${item.seq_id}
                            </td>
                            <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center; font-size: 14px;">
                                ${item.center_position}
                            </td>
                            <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">
                                <div style="display: inline-block; position: relative; width: 80px; height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden;">
                                    <div style="position: absolute; top: 0; left: 0; height: 100%; background: ${isPositive ? '#4CAF50' : '#f44336'}; width: ${probPercent}%;"></div>
                                    <span style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-weight: 600; font-size: 12px;">
                                        ${probPercent}%
                                    </span>
                                </div>
                            </td>
                            <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">
                                <span style="display: inline-block; padding: 4px 8px; border-radius: 10px; font-size: 14px; font-weight: 600; ${isPositive ? 'background: #e8f5e9; color: #2e7d32;' : 'background: #ffebee; color: #c62828;'}">
                                    ${isPositive ? '4mC' : 'Negative'}
                                </span>
                            </td>
                        </tr>`;
                });
                
                if (data.window_details.length > 50) {
                    resultHtml += `
                        <tr>
                            <td colspan="4" style="padding: 10px; text-align: center; font-size: 14px; color: #666;">
                                ... and ${data.window_details.length - 50} more window predictions
                            </td>
                        </tr>`;
                }
                
                resultHtml += `
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>`;
            }
            
            // 添加图例
            resultHtml += `
                <div style="margin-top: 25px; padding: 20px; background: #f8f9fa; border-radius: 6px; border: 1px solid #ddd; font-size: 18px;">
                    <div style="font-weight: bold; color: #0056b3; margin-bottom: 15px; font-size: 20px; text-align: center;">Legend</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">
                        <div style="display: flex; align-items: center; font-size: 18px;">
                            <span style="display: inline-block; width: 15px; height: 15px; background: #ff9800; border-radius: 2px; margin-right: 10px;"></span>
                            <span><strong>Long Sequences</strong> (>41bp): Sliding window analysis with averaging</span>
                        </div>
                        <div style="display: flex; align-items: center; font-size: 18px;">
                            <span style="display: inline-block; width: 15px; height: 15px; background: #2196f3; border-radius: 2px; margin-right: 10px;"></span>
                            <span><strong>41bp Sequences</strong>: Direct prediction</span>
                        </div>
                        <div style="display: flex; align-items: center; font-size: 18px;">
                            <span style="display: inline-block; width: 15px; height: 15px; background: #9c27b0; border-radius: 2px; margin-right: 10px;"></span>
                            <span><strong>Short Sequences</strong> (<41bp): Standardized to 41bp</span>
                        </div>
                    </div>
                </div>`;
        }
        
        resultHtml += `</div>`;
        resultDiv.innerHTML = resultHtml;
        
    })
    .catch(error => {
        console.error('Prediction error:', error);
        resultDiv.innerHTML = `
            <div class="error" style="font-size: 20px; color: #c62828; background: #ffebee; padding: 1.5rem; border-radius: 6px; border-left: 4px solid #c62828; text-align: center;">
                <div style="font-weight: bold; margin-bottom: 10px;">Error Processing Request</div>
                <div style="font-size: 18px;">${error.message || 'Unknown error occurred'}</div>
                <div style="font-size: 16px; margin-top: 10px; color: #666;">
                    Please check your input and try again. If the problem persists, contact the administrator.
                </div>
            </div>`;
    });
});

// 切换窗口详细信息显示/隐藏
function toggleWindowDetails(elementId) {
    const element = document.getElementById(elementId);
    const icon = document.getElementById(elementId + '_icon');
    
    if (element.style.display === 'none' || element.style.display === '') {
        element.style.display = 'block';
        icon.textContent = '🔽';
    } else {
        element.style.display = 'none';
        icon.textContent = '▶️';
    }
}

// 文件上传预览功能
document.getElementById('sequenceFile').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file && (file.type === 'text/plain' || file.name.endsWith('.fasta') || file.name.endsWith('.fa') || file.name.endsWith('.txt'))) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('sequenceInput').value = e.target.result;
        };
        reader.readAsText(file);
    } else if (file) {
        alert('Please upload a valid FASTA file (.fasta, .fa, or .txt)');
        e.target.value = '';
    }
});

// 清空表单功能
document.getElementById('clearButton').addEventListener('click', function() {
    document.getElementById('sequenceInput').value = '';
    document.getElementById('sequenceFile').value = '';
    document.getElementById('result').innerHTML = '';
    cleanupOldDownloadUrls();
});