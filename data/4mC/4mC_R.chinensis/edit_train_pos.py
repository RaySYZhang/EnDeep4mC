def process_sequences():
    input_file = "/home/zqzhangshuyu/Projs/EnDeep4mC-V2/data/4mC/4mC_R.chinensis/train_pos.txt"
    output_file = "/home/zqzhangshuyu/Projs/EnDeep4mC-V2/data/4mC/4mC_R.chinensis/train_pos_cleaned.txt"
    
    new_id_counter = 1
    deleted_found = False
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        while True:
            # 读取序列头行
            header = fin.readline().strip()
            if not header:  # 文件结束
                break
                
            # 读取序列行
            sequence = fin.readline().strip()
            if not sequence:  # 文件结束
                break
            
            # 检查序列是否包含'N'
            if 'N' in sequence:
                print(f"发现异常序列: {header} - 已删除")
                deleted_found = True
                continue
            
            # 写入新序列（使用新编号）
            fout.write(f">P_{new_id_counter}\n")
            fout.write(sequence + "\n")
            new_id_counter += 1
    
    if not deleted_found:
        print("警告: 未发现包含'N'的异常序列")
    
    print(f"处理完成! 新文件已保存至: {output_file}")
    print(f"原始序列数: 1937, 处理后序列数: {new_id_counter-1}")

if __name__ == "__main__":
    process_sequences()